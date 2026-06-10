# CHƯƠNG 1. GIỚI THIỆU

## 1.1. Bối cảnh

Trong những năm gần đây, các nền tảng xem phim trực tuyến và cộng đồng đánh giá phim như Netflix, Disney+, HBO Max, IMDb, Letterboxd hay các dịch vụ xem phim theo yêu cầu phát triển mạnh mẽ. Người dùng có thể tiếp cận một số lượng phim rất lớn, thuộc nhiều thể loại, quốc gia, thời kỳ phát hành và phong cách khác nhau. Sự phong phú này giúp người dùng có nhiều lựa chọn hơn, nhưng đồng thời cũng tạo ra vấn đề quá tải thông tin: người dùng khó tìm được những bộ phim thật sự phù hợp với sở thích cá nhân nếu chỉ duyệt thủ công.

Hệ thống gợi ý phim được xây dựng để giải quyết vấn đề trên. Thay vì yêu cầu người dùng tự tìm kiếm trong hàng nghìn bộ phim, hệ thống phân tích lịch sử đánh giá, tương tác và thông tin nội dung phim để tạo danh sách phim có khả năng phù hợp nhất với từng người dùng. Trong miền phim ảnh, hai nhóm thông tin đặc biệt quan trọng là:

- Dữ liệu hành vi người dùng, ví dụ người dùng từng đánh giá cao hoặc tương tác tích cực với phim nào.
- Siêu dữ liệu phim, ví dụ thể loại, mô tả nội dung, từ khóa, đạo diễn, diễn viên, năm phát hành, độ phổ biến và hình ảnh poster.

Tuy nhiên, chỉ sử dụng một loại tín hiệu thường không đủ. Các phương pháp học từ hành vi người dùng có thể cá nhân hóa tốt khi người dùng có nhiều lịch sử, nhưng gặp khó khăn với người dùng mới, người dùng có ít tương tác hoặc các phim ít người đánh giá. Ngược lại, các phương pháp dựa trên nội dung có thể khai thác siêu dữ liệu để hiểu đặc điểm của phim, nhưng nếu sử dụng đơn lẻ thì thường chưa phản ánh đầy đủ khẩu vị cá nhân của từng người. Vì vậy, đồ án xây dựng một hệ thống gợi ý phim kết hợp nhiều nguồn tín hiệu, gồm lịch sử tương tác, nội dung phim, độ phổ biến và ngữ cảnh phiên sử dụng hiện tại.

Điểm mở rộng quan trọng của hệ thống là khả năng gợi ý theo “gu phiên hiện tại”. Người dùng có thể chọn một vài phim đang quan tâm trong phiên sử dụng, sau đó hệ thống điều chỉnh danh sách gợi ý theo các phim vừa chọn. Cách tiếp cận này giúp hệ thống hoạt động tốt hơn trong trường hợp người dùng mới, người dùng chưa chọn tài khoản, hoặc khi sở thích ngắn hạn của người dùng khác với lịch sử dài hạn.

## 1.2. Bài toán

Đề tài tập trung vào bài toán gợi ý phim cá nhân hóa. Với một người dùng hoặc một phiên sử dụng hiện tại, hệ thống cần xếp hạng các phim chưa xem và trả về danh sách những phim phù hợp nhất.

### Đầu vào

Dữ liệu đầu vào của hệ thống bao gồm:

- Định danh người dùng, nếu người dùng đã có lịch sử trong hệ thống.
- Lịch sử đánh giá hoặc tương tác giữa người dùng và phim.
- Thông tin cơ bản của phim từ MovieLens hoặc Letterboxd, gồm định danh phim, tên phim và thể loại ban đầu.
- Siêu dữ liệu phim từ TMDb, gồm mô tả nội dung, thể loại mở rộng, từ khóa, đạo diễn, diễn viên, biên kịch, hình ảnh poster, năm phát hành, thời lượng, điểm đánh giá cộng đồng, số lượt đánh giá và độ phổ biến.
- Danh sách phim người dùng chọn trong phiên hiện tại để thể hiện gu xem phim ngắn hạn.
- Mức ưu tiên của gu phiên hiện tại so với lịch sử dài hạn của người dùng.
- Số lượng phim cần trả về trong danh sách gợi ý.

### Đầu ra

Kết quả đầu ra là danh sách phim được sắp xếp theo mức độ phù hợp giảm dần. Mỗi phim trong danh sách gợi ý gồm các thông tin:

- Định danh phim trong hệ thống.
- Định danh phim trên TMDb, nếu có.
- Tên phim.
- Điểm phù hợp do hệ thống tính toán.
- Thể loại phim.
- Hình ảnh poster.
- Mô tả nội dung.
- Đạo diễn.
- Diễn viên chính.
- Các nhãn giải thích lý do gợi ý, ví dụ phù hợp với lịch sử đánh giá, cùng thể loại với phim đã chọn trong phiên, hoặc cùng đạo diễn.

### Phát biểu bài toán

Cho tập người dùng, tập phim và tập tương tác đã quan sát được, hệ thống cần học cách ước lượng mức độ phù hợp giữa từng người dùng và từng phim chưa xem. Sau đó, hệ thống loại bỏ các phim đã xuất hiện trong lịch sử huấn luyện của người dùng, sắp xếp các phim còn lại theo điểm phù hợp và chọn ra danh sách các phim nên gợi ý.

Trong dự án này, một tương tác được xem là tích cực khi người dùng đánh giá phim từ 4.0 trên thang 5.0 trở lên. Vì vậy, bài toán được tiếp cận theo hướng xếp hạng phim dựa trên phản hồi ngầm: hệ thống tập trung vào việc chọn và sắp xếp những phim nên gợi ý trước, thay vì chỉ dự đoán chính xác điểm đánh giá tuyệt đối.

## 1.3. Kịch bản ứng dụng

Hệ thống hỗ trợ ba kịch bản sử dụng chính.

### Kịch bản 1: Người dùng đã có lịch sử đánh giá

Người dùng đã có lịch sử tương tác trong MovieLens hoặc Letterboxd. Khi người dùng truy cập hệ thống và chọn tài khoản, hệ thống truy xuất các phim mà người dùng từng đánh giá tích cực. Từ lịch sử này, hệ thống kết hợp nhiều nguồn thông tin:

- Quan hệ giữa người dùng và phim trong dữ liệu tương tác.
- Nội dung và siêu dữ liệu của các phim người dùng đã thích.
- Biểu diễn học được của người dùng và phim.
- Độ phổ biến của phim trong tập dữ liệu.

Các nguồn thông tin này được chuẩn hóa và kết hợp để tạo danh sách phim gợi ý cá nhân hóa. Đây là kịch bản phù hợp với người dùng đã có đủ lịch sử, vì hệ thống có thể học được khẩu vị dài hạn của người dùng.

### Kịch bản 2: Người dùng mới hoặc chưa chọn tài khoản

Trong trường hợp người dùng chưa có lịch sử hoặc đang sử dụng hệ thống ở chế độ khách, hệ thống vẫn có thể gợi ý dựa trên các phim người dùng chọn trong phiên hiện tại. Người dùng chỉ cần thêm một vài phim vào “gu phiên hiện tại”, hệ thống sẽ tạo hồ sơ sở thích tạm thời từ nội dung của các phim đó và tìm các phim tương tự trong catalog.

Kịch bản này giúp giải quyết vấn đề khởi đầu lạnh cho người dùng mới. Thay vì yêu cầu người dùng phải đánh giá nhiều phim trước khi nhận được gợi ý, hệ thống có thể phản hồi ngay dựa trên một vài lựa chọn ban đầu.

### Kịch bản 3: Người dùng muốn điều chỉnh gu xem phim trong phiên hiện tại

Ngay cả khi người dùng đã có lịch sử dài hạn, sở thích trong một phiên xem cụ thể có thể thay đổi. Ví dụ, một người thường xem phim hành động nhưng hôm nay muốn tìm phim lãng mạn hoặc phim tâm lý nhẹ nhàng. Vì vậy, hệ thống cho phép người dùng thêm hoặc bỏ phim khỏi “gu phiên hiện tại” và điều chỉnh mức ưu tiên của gu phiên so với lịch sử cá nhân.

Khi mức ưu tiên gu phiên cao, hệ thống sẽ tập trung hơn vào các phim giống với những phim vừa được chọn. Khi mức ưu tiên thấp, hệ thống sẽ dựa nhiều hơn vào lịch sử dài hạn của người dùng. Cơ chế này giúp trải nghiệm gợi ý linh hoạt hơn, phù hợp với nhu cầu tức thời thay vì chỉ phản ánh thói quen trong quá khứ.

### Kịch bản 4: Hỏi đáp bằng ngôn ngữ tự nhiên

Ngoài gợi ý theo người dùng và theo phiên, hệ thống còn có chatbot tư vấn phim. Người dùng có thể nhập các yêu cầu như muốn xem phim khoa học viễn tưởng về không gian, phim giống một tác phẩm cụ thể, hoặc phim tâm lý có nội dung cảm động. Chatbot truy xuất các phim liên quan từ catalog dựa trên siêu dữ liệu và trả lời bằng tiếng Việt.

Kịch bản này giúp người dùng tìm phim theo nhu cầu tự nhiên hơn, đặc biệt khi người dùng không biết chính xác tên phim hoặc không muốn thao tác qua nhiều bộ lọc.

## 1.4. Mục tiêu

Mục tiêu của đề tài là xây dựng và đánh giá một hệ thống gợi ý phim có khả năng cá nhân hóa danh sách phim cho người dùng, đồng thời có thể demo qua giao diện web và dịch vụ gợi ý.

Cụ thể, đề tài hướng tới các mục tiêu sau:

1. Xây dựng quy trình xử lý dữ liệu từ MovieLens và Letterboxd theo một cấu trúc thống nhất.
2. Làm giàu catalog phim bằng siêu dữ liệu từ TMDb.
3. Chuyển dữ liệu đánh giá thành tập tương tác tích cực dựa trên ngưỡng đánh giá từ 4.0 trở lên.
4. Biểu diễn dữ liệu dưới nhiều dạng: ma trận tương tác người dùng-phim, đồ thị người dùng-phim và vector nội dung phim.
5. Triển khai các phương pháp gợi ý nền tảng dựa trên độ phổ biến, láng giềng gần, phân rã ma trận, học xếp hạng và mô hình tuyến tính cho dữ liệu phản hồi ngầm.
6. Triển khai phương pháp học biểu diễn trên đồ thị người dùng-phim.
7. Triển khai phương pháp gợi ý dựa trên nội dung và mô hình hai tháp học từ siêu dữ liệu phim.
8. Xây dựng phương pháp kết hợp nhiều nguồn điểm, gồm điểm từ lịch sử tương tác, điểm từ nội dung, điểm từ biểu diễn học được và điểm độ phổ biến.
9. Bổ sung cơ chế gợi ý theo gu phiên hiện tại để hỗ trợ người dùng mới và nhu cầu xem phim ngắn hạn.
10. Đánh giá mô hình bằng các chỉ số xếp hạng trong nhóm 10 gợi ý đầu, gồm độ chính xác, độ bao phủ, chất lượng thứ tự xếp hạng và vị trí xuất hiện của gợi ý đúng đầu tiên.
11. So sánh kết quả trên hai tập dữ liệu MovieLens và Letterboxd.
12. Xây dựng hệ thống demo gồm giao diện người dùng, dịch vụ gợi ý và chatbot tư vấn phim.

## 1.5. Phạm vi nghiên cứu

Trong khuôn khổ đồ án, phạm vi nghiên cứu được giới hạn như sau.

### Về dữ liệu

- Sử dụng MovieLens làm tập dữ liệu chuẩn để huấn luyện và đánh giá.
- Sử dụng Letterboxd làm tập dữ liệu thu thập thực tế để đánh giá thêm khả năng tổng quát hóa.
- Sử dụng TMDb để bổ sung siêu dữ liệu phim.
- Chỉ dùng các trường dữ liệu phục vụ bài toán gợi ý phim, không khai thác thông tin nhạy cảm của người dùng.
- Với Letterboxd, thời điểm tương tác dùng trong quá trình chia dữ liệu được tạo ổn định theo từng người dùng, vì thời gian thu thập dữ liệu không phản ánh chính xác thời điểm người dùng xem phim thật.

### Về bài toán

- Tập trung vào bài toán gợi ý danh sách phim phù hợp nhất cho người dùng.
- Ưu tiên chất lượng xếp hạng danh sách gợi ý hơn là dự đoán chính xác điểm đánh giá tuyệt đối.
- Có xét đến gợi ý theo phiên hiện tại, tức danh sách phim người dùng vừa chọn trong quá trình sử dụng hệ thống.
- Không nghiên cứu các bài toán ngoài phạm vi như dự đoán doanh thu, phân loại cảm xúc đánh giá, nhận diện hình ảnh poster hoặc dự báo xu hướng thị trường phim.

### Về phương pháp

Các nhóm phương pháp trong phạm vi đồ án gồm:

- Gợi ý dựa trên độ phổ biến.
- Lọc cộng tác dựa trên hành vi người dùng.
- Phân rã ma trận và học xếp hạng.
- Học biểu diễn trên đồ thị người dùng-phim.
- Gợi ý dựa trên nội dung phim.
- Mô hình hai tháp kết hợp người dùng và nội dung phim.
- Phương pháp kết hợp nhiều nguồn tín hiệu.
- Chatbot tư vấn phim dựa trên truy xuất thông tin từ catalog.

### Về hệ thống

- Hệ thống phục vụ mục đích học tập, nghiên cứu và demo.
- Quá trình gợi ý sử dụng các kết quả huấn luyện đã lưu sẵn, không huấn luyện lại khi người dùng gửi yêu cầu.
- Giao diện demo hỗ trợ tìm kiếm phim, xem chi tiết phim, xem phim tương tự, thêm phim vào gu phiên hiện tại, điều chỉnh mức ưu tiên gu phiên và nhận gợi ý theo người dùng hoặc theo phiên.
- Không đặt mục tiêu triển khai thương mại, chịu tải lớn hoặc huấn luyện thời gian thực ở quy mô sản phẩm.

## 1.6. Đóng góp của đề tài

Đề tài có các đóng góp chính sau.

### Đóng góp về dữ liệu

- Chuẩn hóa dữ liệu MovieLens và Letterboxd về cùng cấu trúc phục vụ huấn luyện.
- Làm giàu thông tin phim bằng siêu dữ liệu từ TMDb, gồm mô tả nội dung, thể loại mở rộng, từ khóa, đạo diễn, diễn viên, poster, năm phát hành và các chỉ số đánh giá cộng đồng.
- Xây dựng catalog phim đã làm giàu để phục vụ cả huấn luyện, gợi ý, giao diện demo và chatbot.

### Đóng góp về phương pháp

- Triển khai và so sánh nhiều nhóm phương pháp gợi ý trên cùng một quy trình đánh giá.
- Xây dựng phương pháp kết hợp tín hiệu từ lịch sử tương tác, nội dung phim, biểu diễn học được và độ phổ biến.
- Sử dụng siêu dữ liệu phim để hỗ trợ người dùng mới, gợi ý theo phiên hiện tại và phim ít tương tác.
- Bổ sung cơ chế điều chỉnh giữa lịch sử dài hạn của người dùng và gu ngắn hạn trong phiên sử dụng.
- Thực hiện so sánh giữa mô hình có sử dụng và không sử dụng siêu dữ liệu TMDb để đánh giá vai trò của thông tin nội dung.

### Đóng góp về thực nghiệm

- Đánh giá mô hình trên cả MovieLens và Letterboxd.
- Sử dụng các chỉ số đánh giá phù hợp với bài toán gợi ý danh sách, đặc biệt là chất lượng của 10 gợi ý đầu tiên.
- Phân tích riêng các nhóm người dùng có ít lịch sử và nhóm phim ít tương tác để hiểu mô hình phù hợp trong từng trường hợp.

### Đóng góp về hệ thống

- Xây dựng quy trình hoàn chỉnh từ thu thập dữ liệu, làm sạch, tạo đặc trưng, huấn luyện, đánh giá đến lưu kết quả phục vụ gợi ý.
- Xây dựng dịch vụ gợi ý để sinh danh sách phim mà không cần huấn luyện lại.
- Xây dựng giao diện demo cho phép tìm kiếm phim, xem chi tiết, xem phim tương tự, thêm phim vào gu phiên hiện tại và nhận gợi ý theo tài khoản hoặc theo phiên.
- Tích hợp chatbot truy xuất siêu dữ liệu phim và trả lời tư vấn bằng tiếng Việt.

## 1.7. Cấu trúc báo cáo

Báo cáo được tổ chức như sau:

- Chương 1 giới thiệu bối cảnh, bài toán, mục tiêu, phạm vi và đóng góp của đề tài.
- Chương 2 trình bày cơ sở lý thuyết về hệ thống gợi ý, lọc cộng tác, gợi ý dựa trên nội dung, học biểu diễn trên đồ thị và phương pháp kết hợp.
- Chương 3 mô tả dữ liệu sử dụng, quy trình thu thập, làm sạch, làm giàu siêu dữ liệu và phân tích khám phá dữ liệu.
- Chương 4 trình bày phương pháp đề xuất, kiến trúc hệ thống, quy trình xử lý dữ liệu và chi tiết từng nhóm mô hình.
- Chương 5 trình bày thực nghiệm, thiết lập đánh giá, kết quả trên MovieLens và Letterboxd, phân tích định lượng và định tính.
- Chương 6 mô tả quá trình xây dựng hệ thống demo, gồm dịch vụ gợi ý, giao diện người dùng, dashboard phân tích dữ liệu và chatbot tư vấn phim.
- Chương 7 trình bày các khó khăn, hạn chế và hướng xử lý trong quá trình thực hiện.
- Chương 8 tổng kết kết quả đạt được và đề xuất hướng phát triển tiếp theo.
