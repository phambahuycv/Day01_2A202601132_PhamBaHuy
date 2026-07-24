# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Khi temperature tăng, phản hồi trở nên đa dạng và sáng tạo hơn nhưng đồng thời cũng kém ổn định. Khoảng 1.2 trở lên chất lượng có thể bắt đầu giảm tùy câu hỏi, và ở 1.8 phản hồi thường bắt đầu kém mạch lạc, có nguy cơ xuất hiện thông tin không chính xác hoặc diễn đạt rời rạc.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Temperature thấp ưu tiên độ chính xác và tính nhất quán, phù hợp với các tác vụ nghiêm túc như pháp lý. Temperature cao ưu tiên tính sáng tạo và đa dạng, phù hợp với các tác vụ sáng tạo như marketing và quảng cáo.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> *Chi phí ước tính mỗi ngày:
Model lớn (gpt-4o): khoảng 200 USD/ngày.
Model nhỏ (gpt-4o-mini): khoảng 12 USD/ngày.
Trường hợp model lớn xứng đáng với chi phí: Khi cần độ chính xác và khả năng suy luận cao, ví dụ phân tích hợp đồng pháp lý, tư vấn chuyên môn hoặc xử lý tài liệu dài, nơi sai sót có thể gây hậu quả lớn.
Trường hợp model nhỏ là lựa chọn đúng: Khi xây dựng chatbot FAQ, hỗ trợ khách hàng hoặc trả lời các câu hỏi đơn giản với lưu lượng lớn, vì chi phí thấp nhưng vẫn đáp ứng tốt nhu cầu.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Hai phản hồi sẽ khác nhau rõ rệt do system prompt định hình vai trò của mô hình. Với system prompt "Bạn là một nhà thơ...", câu trả lời sẽ giàu hình ảnh ví von, giàu cảm xúc, ít hoặc không dùng thuật ngữ kỹ thuật. Ngược lại, với system prompt "Bạn là kỹ sư phần mềm senior...", phản hồi sẽ chính xác, có cấu trúc rõ ràng, sử dụng thuật ngữ chuyên môn và có thể kèm ví dụ code hoặc ví dụ thực tế.

Từ đó có thể thấy system prompt điều khiển được nhiều khía cạnh của phản hồi như vai trò (persona), giọng văn (tone), mức độ kỹ thuật, phong cách diễn đạt, mức độ chi tiết, cách trình bày và loại ví dụ được sử dụng, trong khi vẫn trả lời cùng một câu hỏi của người dùng.
### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> tiktoken đếm token dựa trên cách mô hình mã hóa văn bản nên chính xác hơn nhiều so với cách ước lượng từ. Đối với tiếng Việt, một từ có thể gồm nhiều âm tiết, dấu câu và ký tự Unicode nên số token không tỷ lệ cố định với số từ. Vì vậy, nếu dùng công thức số từ / 0.75 để dự toán ngân sách API, ta thường dự toán thiếu chi phí khi số token thực tế lớn hơn ước lượng. Để tính chi phí chính xác, nên sử dụng count_tokens() thay vì ước lượng thô.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Trong ba ứng dụng, (b) trợ lý giọng nói hưởng lợi nhiều nhất từ streaming vì có thể bắt đầu đọc từng phần phản hồi ngay khi mô hình sinh ra, giúp giảm đáng kể độ trễ mà người dùng cảm nhận và tạo cảm giác hội thoại tự nhiên. (a) Chatbot văn bản cũng được hưởng lợi vì người dùng có thể đọc câu trả lời ngay khi các token đầu tiên xuất hiện thay vì phải chờ toàn bộ phản hồi hoàn thành. Ngược lại, (c) pipeline dịch tài liệu chạy ngầm ban đêm hầu như không cần streaming, vì không có người dùng chờ trực tiếp; điều quan trọng là hoàn thành toàn bộ tài liệu một cách chính xác và ổn định, nên nhận kết quả sau khi xử lý xong là đủ.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Khi API quá tải, nếu tất cả client đều retry sau một khoảng thời gian cố định, chúng sẽ tiếp tục gửi yêu cầu gần như cùng lúc, tạo ra các "đợt sóng" truy cập và khiến máy chủ khó phục hồi. Exponential backoff khắc phục điều này bằng cách tăng dần thời gian chờ sau mỗi lần retry (ví dụ: 1s → 2s → 4s → 8s), giúp giảm tần suất gửi yêu cầu, phân tán tải và tăng khả năng các lần retry sau thành công.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt đề xuất:

"Bạn là một trợ lý AI chuyên hỗ trợ lập trình và học tập. Hãy trả lời chính xác, rõ ràng, ngắn gọn và có cấu trúc. Khi giải thích khái niệm, hãy đưa ví dụ minh họa hoặc đoạn code nếu phù hợp. Nếu không chắc chắn về thông tin, hãy nói rõ giới hạn thay vì suy đoán."

Hai chỗ quan trọng trong prompt:

"Hãy trả lời chính xác, rõ ràng, ngắn gọn và có cấu trúc."
Nếu xóa phần này, câu trả lời có thể trở nên dài dòng, thiếu bố cục hoặc trình bày kém mạch lạc.
"Nếu không chắc chắn về thông tin, hãy nói rõ giới hạn thay vì suy đoán."
Nếu xóa phần này, trợ lý có nhiều khả năng đưa ra câu trả lời mang tính suy đoán hoặc thông tin chưa được kiểm chứng, làm giảm độ tin cậy của phản hồi.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Một tình huống dễ xảy ra là người dùng trao đổi về một dự án lập trình trong nhiều lượt. Ở những lượt đầu, họ đã cung cấp yêu cầu, ngôn ngữ lập trình và các ràng buộc (ví dụ: dùng Python, không sử dụng thư viện ngoài). Sau hơn 4 lượt hội thoại, các thông tin này không còn trong history nên khi người dùng hỏi tiếp "Hãy sửa đoạn code theo yêu cầu ban đầu", trợ lý có thể quên các ràng buộc trước đó và đưa ra lời giải không đúng ngữ cảnh.

Một cách khắc phục là tóm tắt các lượt hội thoại cũ thành một đoạn ngắn chứa các thông tin quan trọng (mục tiêu, yêu cầu, ràng buộc, quyết định đã thống nhất) rồi đưa phần tóm tắt này vào mỗi lần gọi API. Cách này giúp giữ được ngữ cảnh quan trọng mà không làm số lượng token tăng quá nhiều; ngoài ra có thể tăng giới hạn history một cách có chọn lọc đối với những cuộc hội thoại dài hoặc các chủ đề cần ghi nhớ nhiều thông tin.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
