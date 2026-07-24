"""
K3 — Ngày 1: Khám Phá LLM API (9h00–13h00)
AICB-P1: AI Practical Competency Program, Phase 1

Hướng dẫn:
    1. Làm theo LAB_GUIDE.md — mỗi block có các bước chi tiết và checkpoint.
    2. Điền vào tất cả các chỗ đánh dấu TODO.
    3. KHÔNG đổi chữ ký hàm (tên hàm, tham số).
    4. Import OpenAI BÊN TRONG hàm (xem gợi ý) — nếu import ở đầu file,
       các bài test mock sẽ không hoạt động.
    5. Kiểm tra tiến độ:  pytest tests/test_part1.py -v  (từng phần)
       Chấm điểm tổng:    python grade.py
"""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv



# Nạp OPENAI_API_KEY từ file .env (copy .env.example thành .env và dán key vào)
load_dotenv()

# ---------------------------------------------------------------------------
# Bảng giá ước tính (USD / 1K token) — cập nhật nếu giá thay đổi
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

# Tên model có thể đổi qua .env — ví dụ khi dùng NVIDIA NIM miễn phí
# (xem LAB_GUIDE.md, Phụ lục B). Không đặt gì trong .env thì mặc định OpenAI.
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gpt-4o-mini")


# ===========================================================================
# PART 1 — API CƠ BẢN (Block 1: 10h00–10h40)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.1 — Gọi GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi OpenAI Chat Completions API, trả về nội dung phản hồi + độ trễ.

    Args:
        prompt:      Tin nhắn của người dùng.
        model:       Model OpenAI sử dụng (mặc định: gpt-4o).
        temperature: Độ ngẫu nhiên khi lấy mẫu (0.0 – 2.0).
        top_p:       Ngưỡng nucleus sampling.
        max_tokens:  Số token tối đa được sinh ra.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        from openai import OpenAI            # import BÊN TRONG hàm
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # đo thời gian bằng time.time() trước và sau lời gọi API
    """
    # TODO: import OpenAI, tạo client, gọi chat.completions.create,
    #       đo start/end time, trả về (response_text, latency)
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start
    return response.choices[0].message.content, latency


# ---------------------------------------------------------------------------
# Task 1.2 — Gọi GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với model gpt-4o-mini — nhanh hơn và rẻ hơn.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        Tái sử dụng call_openai() với model=OPENAI_MINI_MODEL — 1 dòng code.
    """
    # TODO: gọi call_openai với model=OPENAI_MINI_MODEL
    return call_openai(prompt, model=OPENAI_MINI_MODEL,
                   temperature=temperature, top_p=top_p, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Task 1.3 — So sánh GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Gọi cả hai model với cùng một prompt và trả về dict so sánh.

    Returns:
        Dict với các key:
            - "gpt4o_response":      str
            - "mini_response":       str
            - "gpt4o_latency":       float
            - "mini_latency":        float
            - "gpt4o_cost_estimate": float  (USD ước tính cho phản hồi)

    Gợi ý:
        cost = (len(response.split()) / 0.75) / 1000 \\
               * PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
        (0.75 từ ≈ 1 token — ước lượng thô; Part 2 sẽ tính chính xác hơn)
    """
    # TODO: gọi call_openai và call_openai_mini, ghép dict kết quả
    gpt4o_response, gpt4o_latency = call_openai(prompt, OPENAI_MODEL)
    mini_response, mini_latency = call_openai_mini(prompt)
    gpt4o_cost_estimate = (len(gpt4o_response.split()) / 0.75) / 1000 * PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
    return {
        "gpt4o_response": gpt4o_response,
        "mini_response": mini_response,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": gpt4o_cost_estimate,
    }


# ===========================================================================
# PART 2 — SYSTEM PROMPT & TOKEN (Block 2: 10h40–11h20)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 2.1 — Chat với system prompt (persona)
# ---------------------------------------------------------------------------
def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với MESSAGES gồm 2 phần: system prompt (định hình vai trò/persona
    của model) và user prompt (câu hỏi thật).

    Args:
        system_prompt: Chỉ dẫn vai trò, ví dụ "Bạn là giáo viên tiểu học,
                       giải thích mọi thứ thật đơn giản."
        user_prompt:   Tin nhắn của người dùng.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    """
    # TODO: giống call_openai nhưng messages có thêm phần tử role="system"
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.time() - start
    return response.choices[0].message.content, latency


# ---------------------------------------------------------------------------
# Task 2.2 — Đếm token bằng tiktoken
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    """
    Đếm số token của một đoạn text bằng thư viện tiktoken.

    Args:
        text:  Đoạn text cần đếm.
        model: Model dùng để chọn bộ mã hóa (encoding).

    Returns:
        Số token (int).

    Gợi ý:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))

        tiktoken cần tải bộ mã hóa từ mạng ở lần chạy đầu. Hãy bọc trong
        try/except — nếu lỗi (offline, model lạ), dùng ước lượng dự phòng:
        max(1, len(text) // 4)   (trung bình 1 token ≈ 4 ký tự)
    """
    # TODO: dùng tiktoken để đếm token, có fallback khi lỗi
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Task 2.3 — Ước tính chi phí chính xác
# ---------------------------------------------------------------------------
def estimate_cost(prompt: str, response: str, model: str = OPENAI_MODEL) -> dict:
    """
    Tính chi phí một lượt gọi API dựa trên số token THẬT (đếm bằng
    count_tokens) và bảng giá PRICING_PER_1K_TOKENS — tách riêng chi phí
    input (prompt) và output (response).

    Returns:
        Dict với các key:
            - "input_tokens":  int
            - "output_tokens": int
            - "input_cost":    float  (USD)
            - "output_cost":   float  (USD)
            - "total_cost":    float  (USD)

    Gợi ý:
        pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"])
        input_cost = input_tokens / 1000 * pricing["input"]
        (.get với fallback: model không có trong bảng giá — ví dụ model NIM
         miễn phí — thì lấy giá gpt-4o làm tham chiếu học tập)
    """
    # TODO: đếm token prompt/response, tra bảng giá, trả về dict 5 key
    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(response, model)
    pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"])
    input_cost = input_tokens / 1000 * pricing["input"]
    output_cost = output_tokens / 1000 * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }


# ===========================================================================
# PART 3 — STREAMING & ĐỘ BỀN (Block 3: 11h30–12h10)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 3.1 — Chatbot streaming có lịch sử hội thoại
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Chatbot dòng lệnh tương tác dùng streaming.

    Hành vi:
        - Stream token từ OpenAI ngay khi chúng được sinh ra (in từng chunk).
        - Duy trì 3 lượt hội thoại gần nhất trong history.
        - Gõ 'quit' hoặc 'exit' để thoát.

    Gợi ý:
        - Giữ list `history` gồm các dict {"role": ..., "content": ...}.
        - Dùng stream=True trong client.chat.completions.create() và lặp:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
        - Sau mỗi lượt, thêm phản hồi assistant vào history.
        - Cắt history còn 3 lượt cuối (6 message): history = history[-6:]
    """
    # TODO: vòng lặp while, đọc input, stream phản hồi, duy trì history
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history = []
    while True:
        user_msg = input("Bạn: ")
        if user_msg.strip().lower() in ("quit", "exit"):
            break
        messages = history + [{"role": "user", "content": user_msg}]
        stream = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, stream=True,
        )
        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            reply += delta
            print(delta, end="", flush=True)
        print()  # xuống dòng sau khi in xong phản hồi
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})
        # Cắt history còn 3 lượt cuối
        history = history[-6:]


# ---------------------------------------------------------------------------
# Task 3.2 — Retry với exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Gọi fn(). Nếu ném exception, thử lại tối đa max_retries lần với
    exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Callable không tham số.
        max_retries: Số lần thử lại tối đa.
        base_delay:  Delay ban đầu (giây) trước lần thử lại đầu tiên.

    Returns:
        Giá trị trả về của fn() khi thành công.

    Raises:
        Exception cuối cùng của fn() sau khi hết số lần thử.
    """
    # TODO: vòng lặp retry với exponential backoff
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise                          # hết lượt → ném lỗi cuối cùng ra
            time.sleep(base_delay * (2 ** attempt))


# ===========================================================================
# PART 4 — MINI-PROJECT: TRỢ LÝ CLI HOÀN CHỈNH (Block 4: 12h10–12h50)
# ===========================================================================
def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:
    """
    Trợ lý CLI hoàn chỉnh — ghép mọi thứ bạn đã xây trong Part 1–3.

    Hành vi:
        1. Dùng `persona` làm system prompt cho TOÀN BỘ phiên chat.
        2. Mỗi lượt: đọc tin nhắn qua get_input(); nếu là 'quit'/'exit'
           (không phân biệt hoa thường) → kết thúc phiên.
        3. Gọi API với stream=True, messages = system + history + tin nhắn mới.
           Bọc lời gọi API trong retry_with_backoff để chịu lỗi tạm thời.
        4. In từng chunk khi stream về, ghép lại thành reply hoàn chỉnh.
        5. Cập nhật history (user + assistant), giữ tối đa 3 lượt cuối
           (6 message): history = history[-6:]
        6. Cộng dồn thống kê bằng count_tokens và estimate_cost.
        7. Dừng khi đạt max_turns (nếu được đặt).

    Args:
        persona:   Mô tả vai trò, dùng làm system prompt.
        get_input: Hàm đọc input (mặc định: input). Tham số này giúp
                   test tự động không cần bàn phím thật.
        max_turns: Số lượt tối đa (None = không giới hạn).

    Returns:
        Dict thống kê phiên chat:
            - "num_turns":    int   (số lượt hỏi–đáp đã thực hiện)
            - "total_tokens": int   (tổng token user + assistant)
            - "total_cost":   float (tổng USD ước tính)
            - "history":      list  (history còn lại sau khi cắt, ≤ 6 message)

    Gợi ý khung sườn:
        if get_input is None:
            get_input = input
        history, num_turns, total_tokens, total_cost = [], 0, 0, 0.0
        while True:
            if max_turns is not None and num_turns >= max_turns:
                break
            user_msg = get_input()
            if user_msg.strip().lower() in ("quit", "exit"):
                break
            messages = [{"role": "system", "content": persona}] + history \\
                       + [{"role": "user", "content": user_msg}]
            # stream = retry_with_backoff(lambda: client.chat...create(
            #              model=..., messages=messages, stream=True))
            # reply = ghép các chunk...
            ...
        return {"num_turns": num_turns, "total_tokens": total_tokens,
                "total_cost": total_cost, "history": history}
    """
    # TODO: triển khai theo khung sườn trong docstring
    if get_input is None:
        get_input = input
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history, num_turns, total_tokens, total_cost = [], 0, 0, 0.0
    while True:
        if max_turns is not None and num_turns >= max_turns:
            break
        user_msg = get_input("Bạn: ")
        if user_msg.strip().lower() in ("quit", "exit"):
            break
        messages = [{"role": "system", "content": persona}] + history + [{"role": "user", "content": user_msg}]
        stream = retry_with_backoff(lambda: client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, stream=True))
        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            reply += delta
            print(delta, end="", flush=True)
        print()  # xuống dòng sau khi in xong phản hồi
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})
        history = history[-6:]  # giữ tối đa 3 lượt cuối (6 message)
        num_turns += 1
        total_tokens += count_tokens(user_msg) + count_tokens(reply)
        cost_info = estimate_cost(user_msg, reply)
        total_cost += cost_info["total_cost"]
    return {
        "num_turns": num_turns,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "history": history
    }


# ===========================================================================
# BONUS (không bắt buộc — cho bạn nào xong sớm)
# ===========================================================================
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Chạy compare_models cho từng prompt trong list.

    Returns:
        List các dict — mỗi dict là kết quả compare_models kèm thêm
        key "prompt" chứa prompt gốc.
    """
    # TODO (bonus): lặp qua prompts, gọi compare_models, thêm key "prompt"
    prompt_comparisons = []
    for prompt in prompts:
        result = compare_models(prompt)
        result["prompt"] = prompt
        prompt_comparisons.append(result)
    return prompt_comparisons



def format_comparison_table(results: list[dict]) -> str:
    """
    Định dạng kết quả batch_compare thành bảng text dễ đọc.

    Cột: Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency
    Gợi ý: cắt text dài còn 40 ký tự cho dễ nhìn.
    """
    # TODO (bonus): dựng chuỗi bảng và trả về
    print("Prompt".center(40), "|", "GPT-4o Response".center(40), "|", "Mini Response".center(40), "|", "GPT-4o Latency".center(15), "|", "Mini Latency".center(15))
    for result in results:
        prompt = result["prompt"][:40]
        gpt4o_response = result["gpt4o_response"][:40]
        mini_response = result["mini_response"][:40]
        gpt4o_latency = f"{result['gpt4o_latency']:15.2f}"
        mini_latency = f"{result['mini_latency']:15.2f}"
        print(prompt.ljust(40), "|", gpt4o_response.ljust(40), "|", mini_response.ljust(40), "|", gpt4o_latency, "|", mini_latency)


# ---------------------------------------------------------------------------
# Entry point — demo chạy thật (cần OPENAI_API_KEY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    '''
    print("=== Câu 1.1 — Độ nhạy của temperature ===")
    print("=== So sánh temperature 0.0, 0.5, 1.0 và 1.5 ===")
    prompt = "Hãy kể cho tôi một sự thật thú vị về Việt Nam."

    response_0, _ = call_openai(prompt, temperature=0.0)
    response_05, _ = call_openai(prompt, temperature=0.5)

    response_1, _ = call_openai(prompt, temperature=1.0)
    response_15, _ = call_openai(prompt, temperature=1.5)

    print(f"Temperature 0.0: {response_0}")
    print(f"Temperature 0.5: {response_05}")
    print(f"Temperature 1.0: {response_1}")
    print(f"Temperature 1.5: {response_15}")

    print("\n=== Block 2 — System Prompt & Token ===")
    print("=== Câu 2.1 — Sức mạnh của persona ===")

    system_prompt_1 = "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
    system_prompt_2 = "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

    user_prompt = "Giải thích blockchain là gì?"
    response_system_1, _ = chat_with_system_prompt(system_prompt_1, user_prompt)
    response_system_2, _ = chat_with_system_prompt(system_prompt_2, user_prompt)

    print(f"Persona 1 (giáo viên tiểu học): {response_system_1}")
    print(f"Persona 2 (chuyên gia tài chính): {response_system_2}")

    print("=== Câu 2.2 — Đếm token (tiktoken vs đếm từ) ===")

    sample_vietnamese_text = """
Một sự thật thú vị về Việt Nam là đất nước này có một hệ thống hang động lớn nhất thế giới, đó là hang Sơn Đoòng. Hang Sơn Đoòng nằm trong Vườn quốc gia Phong Nha-Kẻ Bàng, tỉnh Quảng Bình. 
Hang động này được phát hiện vào năm 1991 bởi một người dân địa phương tên là Hồ Khanh, nhưng mãi đến năm 2009, một đoàn thám hiểm người Anh mới chính thức khảo sát và công bố về kích thước khổng lồ của nó. 
Hang Sơn Đoòng có chiều dài hơn 5 km, cao 200 m và rộng 150 m, đủ lớn để chứa cả một tòa nhà chọc trời 40 tầng. Bên trong hang có cả một hệ sinh thái riêng với rừng cây, sông ngầm 
và các loài động thực vật độc đáo.
"""

    manual_count = len(sample_vietnamese_text.split()) / 0.75
    tiktoken_token_count = count_tokens(sample_vietnamese_text)

    print(f"Đếm token (đếm từ): {manual_count}")
    print(f"Đếm token (tiktoken): {tiktoken_token_count}")
    print(f"Chênh lệch: {abs(manual_count - tiktoken_token_count) / min(manual_count, tiktoken_token_count) * 100:.2f}%")
    '''

    print("=== So sánh model ===")
    result = compare_models(
        "Giải thích khác biệt giữa temperature và top_p trong một câu."
    )
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Trợ lý CLI (gõ 'quit' để thoát) ===")
    stats = run_assistant(
        persona="Bạn là trợ giảng thân thiện của khóa AI thực chiến, "
                "trả lời ngắn gọn bằng tiếng Việt, lưu ý giải thích ngắn gọn nếu có thể các thuật ngữ chuyên môn.",
    )
    print("\n--- Thống kê phiên chat ---")
    for key, value in stats.items():
        if key != "history":
            print(f"{key}: {value}")

    print("\n -- Thống kê lịch sử hội thoại (cắt còn 3 lượt cuối) ---")
    user_prompts = stats["history"][::2]  # user messages
    batch_results = batch_compare([msg["content"] for msg in user_prompts])
    format_comparison_table(batch_results)
