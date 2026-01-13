import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llms import LLMs
from prompts import AGENT_PROMPT
from retriever import TopKRetriever
from config import GROQ_API_KEY
import re

class Agent:
    def __init__(self, client: LLMs, system_prompt: str = AGENT_PROMPT) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.messages: list = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def __call__(self, message=""):
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = self.client.generate_content(self.messages)
        return completion
    

def retrieve_more_information(query: str, call_number):
    if call_number == 1:
        return """
    - Chủ đề: lich tiem chung cho be duoi 1 tuoi Lịch tiêm chủng cho bé dưới 1 tuổi theo khuyến cáo của WHO Lịch tiêm chủng cho bé dưới 1 tuổi theo khuyến cáo của WHO bao gồm các loại vắc xin quan trọng mà trẻ cần được tiêm đủ liều, đúng lịch trong những năm tháng đầu đời. Nếu bỏ lỡ một trong những loại vắc xin dưới đây, trẻ có thể mất đi cơ hội được bảo vệ toàn diện và trọn đời bằng vắc xin . Viêm màng não do não mô cầu tuýp B,C Varicella (Hàn Quốc)
    - Chủ đề: u xuong\nCác loại u xương ác tính\n1. Sarcoma xương\nUng thư xương tạo xương là một loại u xác tính, xảy ra khi các tế bào tạo ra khối u ác tính thay vì xương mới. Bệnh u xương ác tính này phần lớn xảy ra ở những vị trí xương đầu gối hoặc xương vai và các vùng xương dài. ( 3 )\nNgười bệnh có những triệu chứng sưng đau quanh khu vực có khối u xương ác tính, cường độ cơn đau cao, có ảnh hưởng đến đi lại, sinh hoạt hằng ngày.\n2. Sarcoma Ewing\nBệnh u xương ác tính, ung thư có tính chất gia đình Ewing Sarcoma thường gặp ở đối tượng thanh thiếu niên và thanh niên. Dù vậy, bệnh Sarcoma Ewing cũng có thể xảy ra ở trẻ em trên 5 tuổi. Tế bào ung thư của Sarcoma Ewing xuất phát từ các hốc tủy, đó là phần hốc xương nơi tủy xương được tạo ra. Ngoài ra Sarcoma Ewing cũng có thể phát triển trong các mô mềm như mỡ, cơ và mạch máu.\n3. Sarcoma sụn\nBệnh Sarcoma sụn, còn được gọi là ung thư sụn, tiến triển từ những khối u xương ác tính phổ biến. Ung thư sụn thuộc loại ung thư nguyên phát, có tính phát triển và di căn, thường xảy ra ở xương chậu, hông và vai. Bệnh gồm 3 giai đoạn bệnh và nguy cơ di căn đến các cơ quan khác khá cao.\n4. Ung thư di căn xương\nUng thư di căn xương là dạng ung thư thứ phát, xảy ra với những trường hợp đang có tế bào ung thư tại cơ quan khác.\nBệnh có thể là biến chứng từ các ung thư như:\n- Ung thư thận\n- Ung thư vú\n- Ung thư tuyến tiền liệt\n- Ung thư phổi\n- Ung thư tuyến giáp\n5. Bệnh đa u tủy\nBệnh đa u tủy là ung thư của tương bào, phá hủy các xương xung quanh bằng cách xâm lấn và gây tổn thương chúng. Bệnh gây ra các tình trạng tổn thương mỡ trong xương, gãy xương, suy thận hoặc nhiễm trùng.\nNhững triệu chứng phổ biến ở bệnh u xương ác tính đa u tủy là đau xương dai dẳng, đặc biệt ở phần lưng và ngực.
    """ 
    elif call_number == 2:
        return """
    - Chủ đề: lich tiem chung cho be duoi 1 tuoi Lịch tiêm chủng cho bé dưới 1 tuổi theo khuyến cáo của WHO Lịch tiêm chủng cho bé dưới 1 tuổi theo khuyến cáo của WHO bao gồm các loại vắc xin quan trọng mà trẻ cần được tiêm đủ liều, đúng lịch trong những năm tháng đầu đời. Nếu bỏ lỡ một trong những loại vắc xin dưới đây, trẻ có thể mất đi cơ hội được bảo vệ toàn diện và trọn đời bằng vắc xin . Viêm màng não do não mô cầu tuýp B,C Varicella (Hàn Quốc)
    - Chủ đề: u xuong\nCác loại u xương ác tính\n1. Sarcoma xương\nUng thư xương tạo xương là một loại u xác tính, xảy ra khi các tế bào tạo ra khối u ác tính thay vì xương mới. Bệnh u xương ác tính này phần lớn xảy ra ở những vị trí xương đầu gối hoặc xương vai và các vùng xương dài. ( 3 )\nNgười bệnh có những triệu chứng sưng đau quanh khu vực có khối u xương ác tính, cường độ cơn đau cao, có ảnh hưởng đến đi lại, sinh hoạt hằng ngày.\n2. Sarcoma Ewing\nBệnh u xương ác tính, ung thư có tính chất gia đình Ewing Sarcoma thường gặp ở đối tượng thanh thiếu niên và thanh niên. Dù vậy, bệnh Sarcoma Ewing cũng có thể xảy ra ở trẻ em trên 5 tuổi. Tế bào ung thư của Sarcoma Ewing xuất phát từ các hốc tủy, đó là phần hốc xương nơi tủy xương được tạo ra. Ngoài ra Sarcoma Ewing cũng có thể phát triển trong các mô mềm như mỡ, cơ và mạch máu.\n3. Sarcoma sụn\nBệnh Sarcoma sụn, còn được gọi là ung thư sụn, tiến triển từ những khối u xương ác tính phổ biến. Ung thư sụn thuộc loại ung thư nguyên phát, có tính phát triển và di căn, thường xảy ra ở xương chậu, hông và vai. Bệnh gồm 3 giai đoạn bệnh và nguy cơ di căn đến các cơ quan khác khá cao.\n4. Ung thư di căn xương\nUng thư di căn xương là dạng ung thư thứ phát, xảy ra với những trường hợp đang có tế bào ung thư tại cơ quan khác.\nBệnh có thể là biến chứng từ các ung thư như:\n- Ung thư thận\n- Ung thư vú\n- Ung thư tuyến tiền liệt\n- Ung thư phổi\n- Ung thư tuyến giáp\n5. Bệnh đa u tủy\nBệnh đa u tủy là ung thư của tương bào, phá hủy các xương xung quanh bằng cách xâm lấn và gây tổn thương chúng. Bệnh gây ra các tình trạng tổn thương mỡ trong xương, gãy xương, suy thận hoặc nhiễm trùng.\nNhững triệu chứng phổ biến ở bệnh u xương ác tính đa u tủy là đau xương dai dẳng, đặc biệt ở phần lưng và ngực.
    - Chủ đề: u xuong\nCác loại u xương lành tính\n1. U xương sụn\nU xương sụn là loại u lành tính phổ biến nhất, chiếm khoảng 40% các trường hợp u xương lành tính. Tình trạng u xương sụn là do sự bất thường tăng trưởng của sụn và xương, cụ thể là tăng trưởng quá nhiều. Chính vì thế mà bệnh thường xảy ra với những người nằm trong độ tuổi phát triển, từ 13 – 25 tuổi. Thống kê của Học viện Phẫu thuật Chỉnh hình Hoa Kỳ (AAOS) cũng kết luận u xương sụn thường rơi vào những trường hợp trẻ em vị thành niên và thanh niên.\nCác khối u ở xương sụn hình thành ở vị trí đầu xương dài như xương cánh tay hoặc xương chân.\n2. U xơ không cốt hóa\nU xơ không cốt hóa là một loại u xương, hình thành do sự tổn thương xơ hóa của xương với triệu chứng cận lâm sàng là tiêu vỏ xương có xuất hiện tổn thương. Đây là loại u lành tính.\nNgười bị u xơ không cốt hóa sẽ có những ổ khuyết khá nhỏ bên trong xương. Những ổ khuyết này sẽ được lấp đầy bằng mô xơ thay vì mô xương như người khỏe mạnh.\n3. U tế bào khổng lồ\nU tế bào khổng lồ còn được gọi là u đại bào hoặc u hủy cốt bào, là một trong những kiểu u lành tính phổ biến. Tuy nhiên, bệnh có thể tiến triển thanh u tế bào khổng lồ ác tính, vì thế yêu cầu người bệnh cần có sự chú tâm điều trị bệnh hơn các u loại u xương lành tính khác.\nU tế bào khổng lồ diễn ra ở vị trí đầu xương dài, thường gặp là đầu trên xương chày, xương đùi, xương quay và đầu dưới xương cùng. Bệnh gồm 3 giai đoạn:\n- U nhỏ lành tính, vỏ xương bị phá hủy.\n- U tăng kích thước, vỏ xương bị tổn thương và mỏng hơn bình thường\n- U tăng sinh đáng kể, ảnh hưởng xấu đến các phần mềm xung quanh. Đồng thời mạch máu cũng tăng sinh nhiều.\n4. U sụn\nU sụn là tình trạng u nang sụn phát triển bên trong tủy xương, có các loại là u nguyên bào sụn, u xơ sụn và u nội sụn. U nội sụn thường xảy ra như dạng u xương lành tính, trong khi u nguyên bào sụn tương đối hiếm gặp .\nU nội sụn có thể xảy ra ở mọi lứa tuổi và đối tượng. Bệnh thường không có các triệu chứng đi kèm, nhưng sự phát triển của khối u sẽ gây sưng và đau tại vị trí tổn thương.\nĐây là một dạng khối u lành tính, tuy nhiên trong trường hợp người bệnh có nhiều khối u nội sụn, nhất là có xảy ra hiện tượng chảy máu mô mềm thì sẽ có nguy cơ mắc ung thư nội sụn cao hơn người khác.\n5. Nang xương phình mạch\nNang xương phình mạch là sự tổn thương nang ở các vùng hành xương của xương dài, phần lớn xảy ra với những người trên 25 tuổi. Tổn thương do nang xương phình mạch có thể kéo dài từ vài tuần đến vài năm trước khi được chẩn đoán.\nCác nang xương bị tổn thương có xu hướng phát triển chậm hơn bình thường, đồng thời xảy ra tình trạng phồng xương.
    """ 
    else:
        return ""

def end_loop(text:str):
    return "end_loop"
    
def loop(agent:Agent, max_iterations=10, query: str = ""):

    tools = ["retrieve_more_information", "end_loop"]
    # tools = ["calculate", "get_planet_mass"]

    next_prompt = query

    i = 0
  
    while i < max_iterations:
        i += 1
        print(f'The iter {i}')
        result = agent(next_prompt)
        print(result)
        print("==="*5)
        
        if "PAUSE" in result or "Action" in result:
            action = re.findall(r"Action: ([a-z_]+): (.+)", result, re.IGNORECASE)
            
            print(f"Action: {action}")

            chosen_tool = action[0][0]
            print(f"Choosen_tool: {chosen_tool}")
            arg = action[0][1]
            
            if chosen_tool.strip().lower() == "end_loop":
                break

            if chosen_tool in tools:
                result_tool = eval(f"{chosen_tool}('{arg}', {i})")
                next_prompt = f"Observation: {result_tool}"

            else:
                next_prompt = "Observation: Tool not found"

            print(next_prompt)
            continue


if __name__ == '__main__':
    client = LLMs(
        type="online",
        model_name="chatgroq",
        api_key=GROQ_API_KEY,
        model_version="llama-3.3-70b-versatile",
        base_url="https://api.groq.com"
    )
    
    agent = Agent(client=client)
    
    query = """
    Tài liệu: Chủ đề: lich tiem chung cho be duoi 1 tuoi Lịch tiêm chủng cho bé dưới 1 tuổi theo khuyến cáo của WHO Lịch tiêm chủng cho bé dưới 1 tuổi theo khuyến cáo của WHO bao gồm các loại vắc xin quan trọng mà trẻ cần được tiêm đủ liều, đúng lịch trong những năm tháng đầu đời. Nếu bỏ lỡ một trong những loại vắc xin dưới đây, trẻ có thể mất đi cơ hội được bảo vệ toàn diện và trọn đời bằng vắc xin . Viêm màng não do não mô cầu tuýp B,C Varicella (Hàn Quốc)
    Câu hỏi: Hãy phân loại u xương ác tính và lành tính
    """
    loop(agent=agent, max_iterations=5, query=query)
    