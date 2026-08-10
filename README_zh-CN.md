# llm-api

[English](README.md)

一个 Rust 库，为 LLM 聊天 API 提供**统一中间表示（IR）**、IR 与多种服务商
API 格式之间的**双向转换**，以及**可插拔的 HTTP 传输层**。请求只需针对 IR
构建一次，调用时再选择上游 API 格式。

为 agent 应用的调用侧而设计：会话历史——包括 thinking 签名、工具调用与服
务商专有数据——都能忠实往返；IR 未建模的内容也始终触手可及。

## 为什么选择 llm-api

- **自由的 JSON 操控。** 任何 IR 都不可能覆盖所有服务商特性，因此每个 IR
  节点都带有按格式命名空间划分的 `extra` 映射，以 JSON Merge Patch 语义
  （RFC 7396）合并进序列化后的请求：在任意层级设置、覆盖乃至**删除**任何
  生成的字段——整个请求、某条消息、某个内容块均可。请求钩子还允许直接原地
  编辑最终 JSON。
- **可插拔的 HTTP 客户端。** 库本身不做任何 IO。任何 HTTP 栈都可以通过一
  个小巧的 `HttpClient` trait 接入；基于 `reqwest` 的默认实现由 feature
  门控（默认开启）。使用 `default-features = false` 可得到纯数据层——没有
  tokio，也没有 TLS。
- **绝不静默丢弃。** 每次转换都返回警告，带有稳定的警告码、固定的严重度
  （`Semantic` 语义级 / `Cosmetic` 修饰级）和 JSON Pointer 位置。strict
  模式会把语义级损失升级为错误——除非你的 `extra` 已显式覆盖了对应路径。
- **忠实往返。** 同服务商的 `格式 → IR → 格式` 转换先规范化、随后幂等；
  未建模的服务商节点（文档块、内置工具调用、可执行代码等）作为 `Opaque`
  值原位保留。唯一的静默表示性损失：显式为 `null` 的未知字段会规范化为
  缺省；其他所有无法恢复的规范化都会产生警告（`docs/design.md` § 1）。将
  agent 历史持久化为 IR JSON 是受支持、受 semver 保障的用法。

## 支持的格式

| 能力 | OpenAI Chat Completions¹ | OpenAI Responses | Anthropic Messages | Google `generateContent` |
|---|---|---|---|---|
| 聊天（非流式 + SSE 流式） | ✓ | ✓ | ✓ | ✓ |
| 工具调用 | ✓ | ✓ | ✓ | ✓ |
| 图片输入（URL / base64 / 文件 id） | URL、base64 | ✓ | ✓ | ✓ |
| thinking（含回放签名） | 纯文本（`reasoning_content`） | ✓ | ✓ | ✓ |
| 结构化输出 | ✓ | ✓ | 仅 JSON Schema | ✓ |
| 模型列表 | ✓ | ✓ | ✓（分页） | ✓（分页） |
| token 计数 | —（无端点） | ✓ | ✓ | ✓ |

¹ 单一实现即覆盖 DeepSeek（`reasoning_content`）等 CC 方言。第三方格式可
通过实现公开的 `ApiFormat` trait 接入。

## 安装

```toml
[dependencies]
llm-api = "0.1"            # 含基于 reqwest 的默认传输层
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }

# 或者只要纯数据层（无 IO、无 tokio、无 TLS）：
# llm-api = { version = "0.1", default-features = false }
```

MSRV：1.88。

## 快速上手

```rust
use std::sync::Arc;

use llm_api::formats::anthropic_messages::AnthropicMessages;
use llm_api::{CallOptions, Client, Message, ProviderConfig, Request};

async fn quick_start() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ProviderConfig::new(
        Arc::new(AnthropicMessages),
        "https://api.anthropic.com/v1",
        "claude-opus-4-6",
    )?
    .with_auth(std::env::var("ANTHROPIC_API_KEY")?);

    let mut request = Request::with_messages(vec![Message::user_text(
        "Which mountain is the tallest on Earth?",
    )])
    .with_system_text("Answer in one sentence.");
    request.max_output_tokens = Some(1024);

    let client = Client::new(reqwest::Client::new());
    let response = client.send(&provider, &request, &CallOptions::default()).await?;

    println!("{}", response.text());
    for warning in &response.warnings {
        // 没有任何内容被静默丢弃：这里能看到映射过程损失了什么。
        eprintln!("[{:?}] {}", warning.severity, warning.message);
    }
    Ok(())
}
```

切换服务商只需换一个 `ProviderConfig`——`Request` 保持不变。
`response.message` 可直接作为历史进入下一次请求；thinking 签名、Responses
的 item id 与 Google 的 `thoughtSignature` 都会经由其命名空间化的 `extra`
自动回流。

## 流式调用

```rust
use futures_util::StreamExt;
use llm_api::{BlockDelta, StreamEvent};

let mut stream = client.stream(&provider, &request, &CallOptions::default()).await?;
while let Some(item) = stream.next().await {
    if let StreamEvent::BlockDelta { delta: BlockDelta::Text(fragment), .. } = item?.event {
        print!("{fragment}");
    }
}
```

也可以一边渲染增量、一边为历史保留完整消息——累加器会把统一的块级事件
（`MessageStart`、`BlockStart`/`BlockDelta`/`BlockStop`、`MessageDelta`、
`MessageStop`）折叠回一个 `Response`：

```rust
let stream = client.stream(&provider, &request, &CallOptions::default()).await?;
let response = stream.collect().await?;
```

未见到协议终止符就中断的流会报告为错误——静默的 EOF 绝不会被当作完整响应
蒙混过关。

## 逃生通道

**`extra`**——每个 IR 节点上按格式命名空间划分的自由 JSON，以 RFC 7396 语
义合并（对象递归合并、数组/标量替换、`null` 删除）：

```rust
use serde_json::json;

// 在 Anthropic 上重新启用手动 thinking 预算（未建模字段）：
request.extra.set(
    llm_api::ids::ANTHROPIC_MESSAGES,
    "thinking",
    json!({"type": "enabled", "budget_tokens": 2048}),
);
// 深度合并进 Google 的 generationConfig；`null` 删除已生成的键：
request.extra.set(
    llm_api::ids::GOOGLE_GENERATE_CONTENT,
    "generationConfig",
    json!({"thinkingConfig": {"thinkingLevel": null, "thinkingBudget": 512}}),
);
```

每个命名空间只在序列化到对应格式时生效，服务商专有数据绝不会泄漏到其他服
务商。从服务商解析到的非 null 未知字段也落入相同命名空间并原样往返。

**钩子**——作用于序列化后 JSON 的闭包，在转换与 strict 检查之后、发送之前
运行：

```rust
use llm_api::RequestHooks;

let hooks = RequestHooks::new()
    .with_on_message(|index, _role, message| {
        if index == 0 {
            message["cache_control"] = json!({"type": "ephemeral"});
        }
        Ok(())
    })
    .with_on_request(|body| {
        body["service_tier"] = json!("flex");
        Ok(())
    });
let opts = CallOptions::default().with_hooks(hooks);
```

## 警告与 strict 模式

转换绝不静默丢数据：每次损失都会产生一个 `ConversionWarning`，带稳定的
`WarningCode`、固定严重度以及指向问题输出的 JSON Pointer。`Semantic` 表示
模型可见的行为可能改变（thinking 块被丢弃、图片来源不受支持）；
`Cosmetic` 表示只损失了调优信息（缓存提示、采样参数）。

```rust
use llm_api::ConvertOptions;

let provider = provider.with_convert(ConvertOptions::default().strict(true));
```

strict 模式下，构建侧任何未被覆盖的语义级警告都会在发起 IO 之前使调用失
败。解析侧警告永远不会让调用失败——响应已经发生并计费——它们通过
`Response::warnings` / `StreamItem::warnings` 上报。

## 纯转换层

每个格式都可以脱离客户端（且不做 IO）单独使用：从 IR 构建服务商 JSON，或
把服务商 JSON 解析回 IR。

```rust
use llm_api::formats::openai_chat_completions::OpenAiChatCompletions;
use llm_api::{ApiFormat, BuildCtx, CallMode, EndpointUrl};

let ctx = BuildCtx::new(
    EndpointUrl::base("https://api.openai.com/v1")?,
    "gpt-5.6",
    CallMode::Unary,
);
let built = OpenAiChatCompletions.build_request(&request, &ctx)?;
// built.url、built.body（字节）、built.headers、built.warnings

// 反向：服务商 JSON -> IR。
let (ir, parse_warnings) = OpenAiChatCompletions.parse_request(&built.body)?;
```

自定义服务商同理：实现 `ApiFormat`（及 `StreamParser`）trait，然后把你的
格式以 `Arc` 交给客户端即可。

## 模型列表与 token 计数

```rust
let models = client.list_models(&provider).await?;   // 自动翻页取尽
let count = client.count_tokens(&provider, &request, &CallOptions::default()).await?;
```

token 数只来自服务商端点（库从不在本地估算）；Chat Completions 没有该端
点，返回 `Error::NotSupported`。各能力可按端点解耦——例如用 Anthropic 格式
聊天、用 OpenAI 格式列模型——通过 `ProviderConfig::with_models_endpoint` /
`with_count_tokens_endpoint` 配置。

## 自定义 HTTP 传输

只需实现一个 trait，整个库就能跑在你的网络栈上：

```rust,ignore
pub trait HttpClient: Send + Sync {
    fn send(
        &self,
        request: http::Request<Bytes>,   // 最终 URL、头、体——原样发送
        auth: Option<AuthHeader>,        // 发送时注入，绝不进入日志
    ) -> Pin<Box<dyn Future<Output = Result<http::Response<BodyStream>, HttpError>> + Send + '_>>;
}
```

API key 与请求分开传递，因此它不会出现在你的代码可能记录日志的
`http::Request` 里；自带的 reqwest 实现会把注入的头标记为敏感。

## 文档

- [`docs/design.md`](docs/design.md)——完整的 v1 设计：IR 形态、各格式映射
  规则、流式模型、错误模型。
- [`docs/impl_contract.md`](docs/impl_contract.md)——叠加在设计之上、约束
  各格式实现的统一决策。
- 本 README 中的所有代码片段都由
  [`tests/readme_examples.rs`](tests/readme_examples.rs) 做编译校验。

## 许可证

MIT。
