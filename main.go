// 飞书机器人 → zcode2api-go 网关 桥接服务
// 长连接模式（WebSocket 收事件，无需公网回调），收到文本消息后调用网关 OpenAI 兼容接口并回复。
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"os/exec"

	"unicode/utf8"

	lark "github.com/larksuite/oapi-sdk-go/v3"
	larkcore "github.com/larksuite/oapi-sdk-go/v3/core"
	"github.com/larksuite/oapi-sdk-go/v3/event/dispatcher"
	larkim "github.com/larksuite/oapi-sdk-go/v3/service/im/v1"
	larkws "github.com/larksuite/oapi-sdk-go/v3/ws"
	"golang.org/x/text/encoding/simplifiedchinese"
)

type Config struct {
	AppID     string
	AppSecret string
	Gateway   string // 聊天模式：OpenAI 兼容端点（完整 URL）
	APIKey    string // 聊天模式：网关接入 Key
	Model     string
	MaxHist   int    // 每 chat 保留的最大历史条数（user+assistant 合计）
	AgentCmd  string // agent 模式命令模板，含 {file} 占位（消息写入临时文件）；空=聊天模式
	WorkDir   string // agent 模式工作目录
	AdminID   string // 管理员 open_id；空=第一个私聊用户自动锁定为管理员
	Timeout   time.Duration
}

func loadConfig() Config {
	c := Config{
		AppID:     os.Getenv("FEISHU_APP_ID"),
		AppSecret: os.Getenv("FEISHU_APP_SECRET"),
		Gateway:   getenv("GATEWAY_URL", "http://127.0.0.1:3011/v1/chat/completions"),
		APIKey:    os.Getenv("GATEWAY_KEY"),
		Model:     getenv("BOT_MODEL", "glm-5.3-flash"),
		AgentCmd:  os.Getenv("AGENT_CMD"),
		WorkDir:   os.Getenv("WORK_DIR"),
		AdminID:   os.Getenv("ADMIN_OPEN_ID"),
	}
	c.MaxHist = 12
	if v := os.Getenv("MAX_HISTORY"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			c.MaxHist = n
		}
	}
	c.Timeout = 15 * time.Minute
	if v := os.Getenv("TASK_TIMEOUT_MIN"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			c.Timeout = time.Duration(n) * time.Minute
		}
	}
	if c.AppID == "" || c.AppSecret == "" {
		log.Fatal("缺少 FEISHU_APP_ID / FEISHU_APP_SECRET 环境变量")
	}
	return c
}

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

// ---------- 每 chat 会话历史（内存，简单截断） ----------

type ChatStore struct {
	mu    sync.Mutex
	chats map[string][]map[string]string
}

func newChatStore() *ChatStore { return &ChatStore{chats: map[string][]map[string]string{}} }

func (s *ChatStore) Get(chatID string) []map[string]string {
	s.mu.Lock()
	defer s.mu.Unlock()
	h := s.chats[chatID]
	out := make([]map[string]string, len(h))
	copy(out, h)
	return out
}

func (s *ChatStore) Append(chatID string, role, content string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	h := append(s.chats[chatID], map[string]string{"role": role, "content": content})
	if len(h) > 12 {
		h = h[len(h)-12:]
	}
	s.chats[chatID] = h
}

func (s *ChatStore) Reset(chatID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.chats, chatID)
}

// ---------- 网关调用 ----------

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

type chatChoice struct {
	Message struct {
		Content string `json:"content"`
	} `json:"message"`
}

type chatResponse struct {
	Choices []chatChoice `json:"choices"`
	Error   *struct {
		Message string `json:"message"`
	} `json:"error"`
}

var httpClient = &http.Client{Timeout: 120 * time.Second}

func callGateway(cfg Config, history []map[string]string, userMsg string) (string, error) {
	msgs := make([]chatMessage, 0, len(history)+1)
	for _, m := range history {
		msgs = append(msgs, chatMessage{Role: m["role"], Content: m["content"]})
	}
	msgs = append(msgs, chatMessage{Role: "user", Content: userMsg})

	body, _ := json.Marshal(chatRequest{Model: cfg.Model, Messages: msgs, Stream: false})
	req, err := http.NewRequest("POST", cfg.Gateway, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))

	var cr chatResponse
	if err := json.Unmarshal(raw, &cr); err != nil {
		return "", fmt.Errorf("网关响应非 JSON (HTTP %d): %s", resp.StatusCode, truncate(string(raw), 200))
	}
	if resp.StatusCode != 200 {
		msg := "unknown"
		if cr.Error != nil && cr.Error.Message != "" {
			msg = cr.Error.Message
		}
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, msg)
	}
	if len(cr.Choices) == 0 {
		return "", fmt.Errorf("网关返回空 choices")
	}
	return cr.Choices[0].Message.Content, nil
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

// ---------- agent 模式：本机执行 CLI ----------

// 全局串行队列：一次只跑一个 agent 任务，避免并发操作机器互相干扰
var (
	taskMu    sync.Mutex
	taskBusy  bool
	taskQueue = make(chan agentTask, 16)
)

type agentTask struct {
	chatID string
	text   string
	client *lark.Client
	cfg    Config
}

func agentWorker() {
	for t := range taskQueue {
		log.Printf("[agent] 开始任务 chat=%s: %s", t.chatID, truncate(t.text, 80))
		ans := runAgent(t.cfg, t.text)
		log.Printf("[agent] 任务完成 chat=%s 回复摘要: %s", t.chatID, truncate(strings.ReplaceAll(ans, "\n", " "), 120))
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		replyText(ctx, t.client, t.chatID, truncate(ans, 4000))
		cancel()
		taskMu.Lock()
		taskBusy = false
		taskMu.Unlock()
	}
}

func runAgent(cfg Config, prompt string) string {
	if cfg.AgentCmd == "" || cfg.WorkDir == "" {
		return "agent 模式未配置完整（AGENT_CMD / WORK_DIR）"
	}
	// 消息从 stdin 管道喂给子进程，不落盘、不经 cmd 重定向
	// （Go exec 会把引号转义成 \"，cmd 不认该转义，< "path" 重定向必报"语法不正确"）
	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "cmd", "/c", cfg.AgentCmd)
	cmd.Dir = cfg.WorkDir
	cmd.Stdin = strings.NewReader(prompt)
	out, err := cmd.CombinedOutput()
	elapsed := "（未超时）"
	if ctx.Err() == context.DeadlineExceeded {
		elapsed = "（已超时中止，输出为部分结果）"
	}
	res := normalizeOutput(out)
	if res == "" {
		if err != nil {
			return fmt.Sprintf("agent 无输出，退出码异常: %v %s", err, elapsed)
		}
		return "agent 执行完成，无输出。" + elapsed
	}
	return res + "\n" + elapsed
}

// normalizeOutput 统一输出编码并清理杂质：
// Windows 下 cmd/控制台程序输出 GBK（非法 UTF-8 字节序列），node 系 CLI 输出 UTF-8；
// 先按 UTF-8 校验，非法则按 GBK 解码；再过滤 CLI 的环境警告行。
func normalizeOutput(out []byte) string {
	b := out
	if !utf8.Valid(b) {
		if dec, err := simplifiedchinese.GBK.NewDecoder().Bytes(b); err == nil {
			b = dec
		}
	}
	lines := strings.Split(string(b), "\n")
	kept := lines[:0]
	for _, ln := range lines {
		t := strings.TrimSpace(ln)
		if t == "" || strings.HasPrefix(t, "[claude-code:") {
			continue
		}
		kept = append(kept, ln)
	}
	return strings.TrimSpace(strings.Join(kept, "\n"))
}

// ---------- 飞书消息 ----------

var atPlaceholder = regexp.MustCompile(`@_user_\d+`)
var cmdReset = regexp.MustCompile(`(?i)^/(reset|重置)\s*$`)

func extractText(content string) string {
	var m struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal([]byte(content), &m); err != nil {
		return ""
	}
	t := strings.TrimSpace(atPlaceholder.ReplaceAllString(m.Text, ""))
	return strings.TrimSpace(t)
}

func replyText(ctx context.Context, client *lark.Client, chatID, text string) {
	content, _ := json.Marshal(map[string]string{"text": text})
	req := larkim.NewCreateMessageReqBuilder().
		ReceiveIdType("chat_id").
		Body(&larkim.CreateMessageReqBody{
			ReceiveId: &chatID,
			MsgType:   ptr("text"),
			Content:   ptr(string(content)),
		}).
		Build()
	resp, err := client.Im.Message.Create(ctx, req)
	if err != nil {
		log.Printf("[reply] 调用失败: %v", err)
		return
	}
	if !resp.Success() {
		log.Printf("[reply] 业务失败: code=%d msg=%s", resp.Code, resp.Msg)
	}
}

func ptr[T any](v T) *T { return &v }

// ---------- main ----------

func main() {
	cfg := loadConfig()
	store := newChatStore()

	if cfg.AgentCmd != "" {
		go agentWorker()
	}

	client := lark.NewClient(cfg.AppID, cfg.AppSecret)

	dispatcher := dispatcher.NewEventDispatcher("", "")
	// 用户进入机器人聊天界面会推 bot_p2p_chat_entered_v1，注册空 handler 避免 not found 报错刷日志
	dispatcher.OnP2ChatAccessEventBotP2pChatEnteredV1(func(ctx context.Context, event *larkim.P2ChatAccessEventBotP2pChatEnteredV1) error {
		return nil
	})
	dispatcher.OnP2MessageReceiveV1(func(ctx context.Context, event *larkim.P2MessageReceiveV1) error {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("[event] panic 已兜底: %v", r)
			}
		}()
		if event.Event == nil || event.Event.Message == nil {
			return nil
		}
		msg := event.Event.Message

		// 只处理文本消息；忽略机器人自身/其它机器人消息，防自循环
		if msg.MessageType == nil || *msg.MessageType != "text" {
			return nil
		}
		if event.Event.Sender != nil && event.Event.Sender.SenderType != nil && *event.Event.Sender.SenderType == "bot" {
			return nil
		}

		chatID := deref(msg.ChatId)
		text := extractText(deref(msg.Content))
		if text == "" {
			return nil
		}

		// 发件人 open_id（agent 模式下管理员锁定用）
		senderOpenID := ""
		if event.Event.Sender != nil && event.Event.Sender.SenderId != nil {
			senderOpenID = deref(event.Event.Sender.SenderId.OpenId)
		}

		// agent 模式：锁定管理员；未锁定时第一个私聊用户自动成为管理员
		if cfg.AgentCmd != "" {
			if cfg.AdminID == "" {
				if deref(msg.ChatType) != "p2p" {
					return nil // 未锁定管理员前忽略群消息
				}
				cfg.AdminID = senderOpenID
				log.Printf("[auth] 管理员已锁定: %s", senderOpenID)
				replyText(ctx, client, chatID, "已将你锁定为管理员，之后的指令我会直接在这台机器上执行。")
				return nil
			}
			if senderOpenID != cfg.AdminID {
				log.Printf("[auth] 拒绝非管理员: %s", senderOpenID)
				return nil // 静默忽略
			}
		}

		if cmdReset.MatchString(text) {
			store.Reset(chatID)
			replyText(ctx, client, chatID, "已清空本会话上下文。")
			return nil
		}

		// agent 模式：入串行队列，先回执
		if cfg.AgentCmd != "" {
			taskMu.Lock()
			busy := taskBusy
			taskBusy = true
			taskMu.Unlock()
			if busy {
				select {
				case taskQueue <- agentTask{chatID: chatID, text: text, client: client, cfg: cfg}:
					replyText(ctx, client, chatID, "有任务在跑，已排队（前面还有 "+strconv.Itoa(len(taskQueue))+" 个）。")
				default:
					taskMu.Lock()
					taskBusy = false
					taskMu.Unlock()
					replyText(ctx, client, chatID, "队列已满，稍后再发。")
				}
				return nil
			}
			replyText(ctx, client, chatID, "收到，开始执行："+truncate(text, 200))
			taskQueue <- agentTask{chatID: chatID, text: text, client: client, cfg: cfg}
			return nil
		}

		// 聊天模式：每条消息独立 goroutine 处理，不阻塞事件循环
		go func(text, chatID string) {
			bgCtx, cancel := context.WithTimeout(context.Background(), 150*time.Second)
			defer cancel()

			history := store.Get(chatID)
			store.Append(chatID, "user", text)

			ans, err := callGateway(cfg, history, text)
			if err != nil {
				log.Printf("[gateway] chat=%s err=%v", chatID, err)
				replyText(bgCtx, client, chatID, "网关调用失败："+truncate(err.Error(), 300))
				return
			}
			store.Append(chatID, "assistant", ans)
			replyText(bgCtx, client, chatID, truncate(ans, 4000))
		}(text, chatID)

		return nil
	})

	wsClient := larkws.NewClient(
		cfg.AppID,
		cfg.AppSecret,
		larkws.WithEventHandler(dispatcher),
		larkws.WithDomain(lark.FeishuBaseUrl),
		larkws.WithLogLevel(larkcore.LogLevelInfo),
	)

	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		log.Println("收到退出信号")
		os.Exit(0)
	}()

	log.Printf("飞书机器人启动: mode=%s", map[bool]string{true: "agent", false: "chat"}[cfg.AgentCmd != ""])
	if err := wsClient.Start(context.Background()); err != nil {
		log.Fatalf("WebSocket 连接退出: %v", err)
	}
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}
