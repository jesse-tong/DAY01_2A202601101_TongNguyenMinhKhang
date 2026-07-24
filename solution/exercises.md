# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Sau khi cho AI trả lời với 4 temperature trên vài lần, tôi nhận thấy rằng temperature càng cao, mỗi lần trả lời thì càng ngẫu nhiên hơn (có thể dẫn các nguồn khác nhau), với temperature cao (1.0 hoặc 1.5) thì có thể thay đổi cả chủ đề câu hỏi (ví dụ như trả lời về cà phê trứng thay vì hang Sơn Đoòng). Với temperature thấp, thì thay đổi câu trả lời giữa các lần ít hơn.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Với chatbot hỗ trợ khách hàng, tôi sẽ đặt temperature ở mức 0.5 tới 0.7. Điều này giúp bot trả lời nhất quán và tập trung vào đúng chủ đề hơn (với các người dùng khác nhau), đồng thời vẫn giữ được một chút linh hoạt để xử lý các tình huống không mong đợi.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> *Theo giá API hiện tại cho 1 triệu token thì GPT-4o output giá là $10.00, còn GPT-4o-mini giá là $0.6. Như vậy, với workload này, GPT-4o đắt hơn GPT-4o-mini khoảng 16.7 lần. Một trường hợp GPT-4o xứng đáng với chi phí là khi cần độ chính xác cao và xử lý các yêu cầu phức tạp mà không yêu cầu độ trễ thấp, với output không quá dài, chẳng hạn đọc và phân tích, trả lời câu hỏi trong nhiều tài liệu kỹ thuật.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Hai phản hồi khác nhau rõ rệt về từ vựng và cách trình bày. Với system prompt là giáo viên tiểu học, câu trả lời trình bày thân mật hơn, so sánh blockchain với việc thêm các hộp vào chuỗi, và so sánh việc nhiều người cùng xem và kiểm tra các hộp này (thay vì sử dụng thuật ngữ consensus), trong khi đó với system prompt là chuyên gia tài chính, câu trả lời được trình bày giống định nghĩa trong tài liệu chuyên môn. Câu trả lời với prompt giáo viên tiểu học không sử dụng thuật ngữ chuyên môn và tiếng Anh, đồng thời ít có các từ ngữ phức tạp, trong khi đó prompt chuyên gia tài chính sử dụng các thuật ngữ kỹ thuật phức tạp và trả lời chi tiết về kỹ thuật và các thuật ngữ CNTT và tài chính (trả lời kiểu người hỏi đã biết các thuật ngữ như hàm băm mật mã).*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *Số token với ước lượng: 183, số token với tiktoken: 190, chênh lệch: 3.82%. Tiếng Việt thường tốn nhiều token hơn tiếng Anh cùng độ dài vì tiếng Việt có nhiều từ ghép và các từ có dấu nên thường một từ tiếng Việt sẽ được tách thành nhiều token, trong khi tiếng Anh ít từ ghép hơn và không có thanh điệu nên một từ thường chỉ ứng với một token khi tokenize.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming quan trọng nhất khi cần cung cấp phản hồi cuối cùng nhanh chóng cho người dùng trong thời gian thực, ví dụ như trong các ứng dụng chat. Non-streaming lại phù hợp hơn khi cần xử lý toàn bộ phản hồi trước khi hiển thị và trong các tình huống xử lý cần nhiều thời gian (như xử lý và phân tích nhiều tài liệu), hoặc response giữa các bước xử lý (chẳng hạn như response các model ở các bước trung gian như gọi tool trong AI Agent).*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *So với delay cố định, exponential backoff giúp phân tán các yêu cầu retry ra trong thời gian dài hơn, giảm tải cho server và tránh tình trạng "thundering herd" khi hàng nghìn client cùng gửi yêu cầu cùng lúc. Nếu hàng nghìn client cùng retry với delay cố định, sẽ gây ra tình trạng các client cùng retry cùng lúc nhiều lần, làm tăng tải cho server và tốn tài nguyên client do server chỉ có thể xử lý một số client mỗi lần các client cùng retry cùng một lúc.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *System prompt tôi chọn là: 'Bạn là trợ giảng thân thiện của khóa AI thực chiến, trả lời ngắn gọn bằng tiếng Việt, lưu ý giải thích ngắn gọn nếu có thể các thuật ngữ chuyên môn'. System prompt này trả lời theo phong cách thân thiện là vì trợ lý này hướng tới người dùng cuối là học viên AI thực chiến nên tông giọng không nên quá nghiêm túc, trả lời bằng tiếng Việt là vì người dùng trợ lý này là người Việt Nam, giải thích các thuật ngữ chuyên môn là vì học viên có thể chưa biết một số thuật ngữ trong câu trả lời, đặc biệt khi câu hỏi thuộc lĩnh vực hoặc chuyên ngành khác.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Trợ lý hiện tại có hạn chế là không có bộ nhớ dài hạn, dẫn đến việc không thể ghi nhớ thông tin từ các cuộc trò chuyện trước. Một cải tiến cụ thể là thêm chức năng lưu trữ lịch sử hội thoại trong một cơ sở dữ liệu và tool cho trợ lý có thể tìm kiểm trong lịch sử hội thoại, cho phép trợ lý truy cập và sử dụng thông tin từ các cuộc trò chuyện trước, đồng thời thêm một tool để tóm tắt lịch sử trò chuyện khi history đạt ngưỡng nhất định, để hạn chế tình trạng trợ lý quên yêu cầu ban đầu của người dùng và công việc trợ lý đang thực hiện trước đó trong cuộc trò chuyện dài.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
