# SSE Streaming Progress — Design Spec

## Purpose

Add **runtime progress visibility** to the frontend so users can see which step the
backend is currently executing (image analysis → routing → searching → generating),
while fully complying with the competition API specification in README.md.

## Constraints (from README)

- `/chat` response body MUST follow the fixed schema:
  `{"code": 0, "msg": "success", "data": {"answer", "session_id", "timestamp", "returned_images"}}`
- Request body already defines `stream: bool = false` — this field is gated for us.
- When `stream: false` (default), behavior MUST be identical to the current
  synchronous JSON response — this is the path used for official scoring.
- When `stream: true`, the endpoint returns SSE with progress events followed by a
  final `result` event whose `data` payload matches the standard schema exactly.

## Architecture

```
Request (stream: true)
       │
       ▼
 main.py: /chat 端点
       │
       ├─ stream=false → 原 logic，返回 JSON（评审路径）
       │
       └─ stream=true  → StreamingResponse (SSE)
                              │
                              ▼
                    answer.py: generate_with_progress() 生成器
                              │
                              │ yield {"type": "step", ...}  进度事件
                              │ yield {"type": "result",...}  最终结果
                              ▼
                         index.html: EventSource → 进度条 UI → 渲染答案
```

## Backend Changes

### `answer.py` — new generator function

```
generate_with_progress(question, images, session_id) -> Generator[dict]

Yields in order:
  1. {"type": "step", "step": "routing",  "label": "正在分析问题意图..."}
  2. {"type": "step", "step": "searching","label": "正在检索产品手册..."}
     (skipped when ifanswer=True — direct answer)
  3. {"type": "step", "step": "generating","label": "正在生成回答..."}
     (skipped when ifanswer=True — merged with routing)
  4. {"type": "result", "data": {answer, session_id, timestamp, returned_images}}
```

Steps that are conditional:
- `vision` step: only emitted when `len(images) > 0`
- `searching` + `generating`: skipped when routing decides to answer directly

### `main.py` — SSE branch

```python
async def chat_endpoint(request: ChatRequest, ...):
    if request.stream:
        return StreamingResponse(
            sse_wrapper(request),
            media_type="text/event-stream"
        )
    # else: existing sync code path, unchanged

async def sse_wrapper(request):
    for event in generate_with_progress(...):
        yield f"event: {event['type']}\ndata: {json.dumps(event.get('data') or event)}\n\n"
```

### `prompt.py` — no changes needed

Language-aware prompts from the prior optimization already handle routing correctly.

## Frontend Changes

### `index.html` — progress indicator component

Seven predefined step labels (mapped from `step` field):

| step key | Display text (CN) |
|----------|-------------------|
| `memory` | 正在加载历史对话... |
| `vision` | 正在解析图片信息... |
| `routing` | 正在分析问题意图... |
| `searching` | 正在检索产品手册... |
| `generating` | 正在生成回答... |
| `done` | (terminal — show checkmark) |

### UI behavior

1. User sends message → append user bubble + progress widget
2. Each SSE `step` event → mark current step as `active` (blue pulse), previous steps as `done` (green check)
3. SSE `result` event → remove progress widget, render AI answer bubble with images
4. On error / SSE disconnect → replace progress widget with error message

### Component states

```
○ 正在加载历史对话...          ← pending (gray)
◉ 正在分析问题意图...          ← active (blue, pulsing)
✓ 正在分析问题意图...          ← done (green check)
```

### Fallback

If browser doesn't support SSE or `EventSource` errors out, fall back to
`stream: false` request — the existing loading dots still work.

## Data Flow Diagram

```
User clicks Send
  → appendUserMessage()
  → appendProgressWidget()
  → fetch(..., {stream: true})
       │
       ▼ ReadableStream
  ┌─ "event: step\ndata: {step: routing, label: ...}"
  │   → updateProgressWidget("routing")
  │
  ├─ "event: step\ndata: {step: searching, label: ...}"
  │   → updateProgressWidget("searching")
  │
  ├─ "event: step\ndata: {step: generating, label: ...}"
  │   → updateProgressWidget("generating")
  │
  └─ "event: result\ndata: {answer, session_id, ...}"
      → removeProgressWidget()
      → appendAIMessage(answer, returned_images)
```

## Error Handling

| Scenario | Behavior |
|----------|----------|
| SSE connection drops mid-stream | Show "连接中断" error, keep progress widget as-is |
| Backend exception during streaming | Yield `{"type": "error", "data": {"msg": "..."}}`, frontend shows error |
| `stream: false` path | Unchanged — existing error handling |
| Browser no SSE support | Fall back to `stream: false` path |

## Non-Goals

- Real token-level streaming (TTFB streaming) — the final answer still arrives as a
  complete block. This can be added later without changing the SSE protocol.
- WebSocket — overkill for one-directional progress events.
- Persisting progress state across page reloads.

## Files Touched

| File | Change |
|------|--------|
| `code/answer.py` | Add `generate_with_progress()` generator (~60 lines) |
| `code/main.py` | Add SSE branch in `/chat` endpoint, `sse_wrapper()` helper (~25 lines) |
| `index.html` | Add progress widget, SSE reader, step label map (~80 lines) |
