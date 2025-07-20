# test_api.py
import requests

def test_summarize():
    url = "http://127.0.0.1:8000/api/v1/summarize"
    
    data = {
        "content": """Sáng 18/7, phát biểu khai mạc hội nghị lần thứ 12 Ban Chấp hành Trung ương Đảng khóa 13, Tổng Bí thư Tô Lâm cho biết hội nghị được tổ chức sớm gần ba tháng so với kế hoạch, thể hiện tinh thần chủ động, khẩn trương, trách nhiệm cao trong chuẩn bị toàn diện cho Đại hội đại biểu toàn quốc lần thứ 14.

        Tổng Bí thư gửi lời cảm ơn toàn Đảng đã đoàn kết, đồng lòng cùng nhân dân cả nước triển khai thành công mô hình chính quyền địa phương hai cấp từ ngày 1/7. Ông nhấn mạnh đây là bước chuyển mang tính lịch sử, mở ra chương mới trong công cuộc xây dựng một thiết chế quản trị hiện đại, liêm chính, tinh gọn và hướng tới người dân.

        Người đứng đầu Đảng cho biết hội nghị diễn ra trong bối cảnh cả hệ thống chính trị đang chuyển trạng thái từ vừa chạy vừa xếp hàng sang hàng thẳng, lối thông, đồng lòng cùng tiến. Từ Trung ương đến 34 tỉnh, thành phố và hơn 3.300 xã, phường, đặc khu, bộ máy hành chính các cấp đang vận hành theo hướng tinh gọn, hạn chế trung gian, xóa bỏ trùng lắp chức năng, nâng cao chất lượng quản trị, gần dân, sát dân, phục vụ nhân dân tốt hơn.

        Hội nghị Trung ương 12 sẽ tập trung thảo luận ba nhóm nội dung lớn: công tác chuẩn bị Đại hội 14; tạo cơ sở chính trị, pháp lý cho mục tiêu tiếp tục cải cách, đổi mới; và công tác cán bộ. Trong đó, vấn đề nhân sự cấp cao được xác định là nội dung đặc biệt hệ trọng, ảnh hưởng trực tiếp đến thành bại của toàn bộ nhiệm kỳ tới.

        Trung ương sẽ xem xét bổ sung quy hoạch Ban Chấp hành Trung ương, Bộ Chính trị, Ban Bí thư khóa 14; phương hướng công tác cán bộ và các nội dung thuộc thẩm quyền. Nhân sự phải có bản lĩnh chính trị vững vàng, tư duy đổi mới, đạo đức trong sáng, hành động quyết liệt vì tập thể, vì nhân dân, đặt lợi ích quốc gia, dân tộc lên trên hết, trước hết, Tổng Bí thư nói, nhấn mạnh việc tuân thủ lời dạy của Chủ tịch Hồ Chí Minh cán bộ là cái gốc của mọi công việc.""",
        "max_length": 100
    }
    
    print("🚀 Testing summarization API...")
    print(f"📊 URL: {url}")
    print(f"📊 Content length: {len(data['content'])} chars")
    
    try:
        response = requests.post(url, json=data, timeout=600)
        print(f"✅ Status Code: {response.status_code}")
        print(f"📝 Response Headers: {dict(response.headers)}")
        print(f"📝 Response Text: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"🎯 Success!")
            print(f"📄 Summary: {result['data'][0]['summary']}")
        else:
            print(f"❌ Error Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running on port 8000?")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_summarize()