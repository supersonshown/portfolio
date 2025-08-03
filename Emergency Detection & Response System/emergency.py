

import os
import requests
import pandas as pd
import torch
import folium
import polyline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from geopy.geocoders import Nominatim
from openai import OpenAI
from warnings import filterwarnings

filterwarnings('ignore')

class EmergencyAssistant:
    def __init__(self, save_directory, audio_path, hospital_data):
        self.save_directory = save_directory
        self.audio_path = audio_path
        self.hospital_data = pd.read_csv(hospital_data)
        self.client = OpenAI()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(save_directory).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(save_directory)
        # Naver API 키 설정
        self.naver_client_id = 'sb7vih3g35'
        self.naver_client_secret = '4hp1aMpvQ9mRmhQ6h9U1xd5hV96x2so6Iz74vYiU'

    def audio_to_text(self, filename):
        audio_file = open(os.path.join(self.audio_path, filename), "rb")
        transcript = self.client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            language="ko",
            response_format="text",
        )
        return transcript

    def text_summary(self, input_text):
        system_role = '''당신은 신문기사에서 핵심을 요약하는 어시스턴트입니다.
        응답은 다음의 형식을 지켜주세요:
        {"summary": "텍스트 요약"}
        '''
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": input_text},
            ]
        )
        return response.choices[0].message.content

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits.softmax(dim=1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        return predicted_class, logits

    def get_lat_long(self, address):
        geolocator = Nominatim(user_agent="emergency_assistant")
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            raise ValueError("주소를 찾을 수 없습니다.")

    def get_dist(self, start_lat, start_lng, dest_lat, dest_lng):
        url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": self.naver_client_id,
            "X-NCP-APIGW-API-KEY": self.naver_client_secret
        }
        params = {
            "start": f"{start_lng},{start_lat}",
            "goal": f"{dest_lng},{dest_lat}",
            "option": "trafast"
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            try:
                return data['route']['trafast'][0]['summary']['distance']
            except (KeyError, IndexError):
                raise ValueError("올바르지 않은 응답 데이터 형식입니다.")
        else:
            raise ConnectionError(f"API 요청 실패: {response.status_code}, {response.text}")

    def process_audio_files(self):
        file_names = [f for f in os.listdir(self.audio_path) if os.path.isfile(os.path.join(self.audio_path, f))]
        results = []
        for filename in file_names:
            text = self.audio_to_text(filename)
            summary = self.text_summary(text)
            predicted_class, _ = self.predict(summary)
            results.append((filename, text, summary, predicted_class))
        return results

    def handle_emergency(self, address):
        lat, lng = self.get_lat_long(address)
        recommendations = self.recommend_hospital(lat, lng)
        return recommendations

    def calculate_road_distance(self, start_coords, end_coords):
        """
        네이버 API를 사용하여 실제 도로를 따라가는 경로의 거리와 시간을 계산합니다.
        """
        url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": self.naver_client_id,
            "X-NCP-APIGW-API-KEY": self.naver_client_secret
        }
        params = {
            "start": f"{start_coords[1]},{start_coords[0]}",  # 경도,위도 순서
            "goal": f"{end_coords[1]},{end_coords[0]}",
            "option": "trafast"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                route = data['route']['trafast'][0]
                # 거리를 미터에서 킬로미터로 변환
                distance_km = route['summary']['distance'] / 1000
                # 시간을 초에서 분으로 변환
                duration_min = route['summary']['duration'] / 60000  # 밀리초를 분으로 변환
                # 경로 좌표 추출
                path_coordinates = route['path']
                
                return distance_km, duration_min, path_coordinates
            return None, None, None
        except Exception as e:
            print(f"거리 계산 중 오류 발생: {e}")
            return None, None, None

    def visualize_hospitals_with_route(self, user_coords, hospitals):
        # 사용자 위치를 중심으로 지도를 생성
        m = folium.Map(location=user_coords, zoom_start=13)
        
        # 사용자 위치 마커 추가
        folium.Marker(
            location=user_coords,
            tooltip="사용자 위치",
            icon=folium.Icon(color="blue")
        ).add_to(m)
        
        # 병원 위치 마커 추가 (추천 병원들)
        for _, row in hospitals.iterrows():
            # 실제 도로 경로에 따른 거리 계산
            hospital_coords = (row['위도'], row['경도'])
            road_distance, duration, path_coordinates = self.calculate_road_distance(user_coords, hospital_coords)
            
            # 병원 마커 추가
            if road_distance is not None:
                hospital_info = (
                    f"<div style='font-size: 12px;'>"
                    f"<b>{row['name']}</b><br>"
                    f"도로 거리: {road_distance:.1f}km<br>"
                    f"예상 소요시간: {duration:.0f}분"
                    f"</div>"
                )
            else:
                hospital_info = f"{row['name']}\n거리 계산 불가"
            
            folium.Marker(
                location=hospital_coords,
                popup=folium.Popup(hospital_info, max_width=200),
                tooltip=row['name'],
                icon=folium.Icon(color="red")
            ).add_to(m)
            
            # 경로 표시 추가
            if path_coordinates:
                # 네이버 API에서 받은 좌표를 folium 형식으로 변환
                route_coordinates = [[coord[1], coord[0]] for coord in path_coordinates]
                
                # 경로를 지도에 추가
                folium.PolyLine(
                    locations=route_coordinates,
                    weight=3,
                    color='red',
                    opacity=0.8,
                    tooltip=f'거리: {road_distance:.1f}km, 소요시간: {duration:.0f}분'
                ).add_to(m)
                
                # 경로의 중간 지점에 거리 정보 표시
                mid_point_index = len(route_coordinates) // 2
                mid_point = route_coordinates[mid_point_index]
                
                folium.Popup(
                    f'거리: {road_distance:.1f}km<br>소요시간: {duration:.0f}분',
                    max_width=200
                ).add_to(folium.RegularPolygonMarker(
                    location=mid_point,
                    number_of_sides=4,
                    radius=0,
                    weight=0,
                    fill=False
                ).add_to(m))
        
        return m

    def recommend_hospital(self, lat, lng, alpha=0.1):
        """
        주변 병원을 추천하고 실제 도로 거리를 계산합니다.
        """
        target_hospitals = self.hospital_data[
            (self.hospital_data['위도'] >= lat - alpha) & 
            (self.hospital_data['위도'] <= lat + alpha) &
            (self.hospital_data['경도'] >= lng - alpha) & 
            (self.hospital_data['경도'] <= lng + alpha)
        ]
        
        recommendations = []
        for _, hospital in target_hospitals.iterrows():
            distance, duration, _ = self.calculate_road_distance(
                (lat, lng), 
                (hospital['위도'], hospital['경도'])
            )
            if distance is not None:
                recommendations.append({
                    'name': hospital['병원이름'],
                    'address': hospital['주소'],
                    '위도': hospital['위도'],
                    '경도': hospital['경도'],
                    'distance': distance * 1000,  # km를 m로 변환
                    'duration': duration
                })
        
        # 거리순으로 정렬
        recommendations.sort(key=lambda x: x['distance'])
        return pd.DataFrame(recommendations).head(3)

    def save_map(self, map_obj, filename="hospital_routes.html"):
        """
        지도를 HTML 파일로 저장합니다.
        """
        map_obj.save(filename)


if __name__ == "__main__":
    # 경로 설정
    path = '/content/drive/MyDrive/kt/projects_kt/project_6_2/6_2/'
    save_directory = path + 'fine_tuned_bert_v2'
    audio_path = path + "audio/"
    hospital_data = path + "응급실 정보.csv"
    
    # OpenAI API 키 설정
    def load_file(filepath):
        with open(filepath, 'r') as file:
            return file.read().strip()
    
    openai.api_key = load_file(path + 'api_key.txt')
    os.environ['OPENAI_API_KEY'] = openai.api_key
    
    # EmergencyAssistant 인스턴스 생성
    assistant = EmergencyAssistant(save_directory, audio_path, hospital_data)
    
    try:
        # Audio 파일 처리
        print("음성 파일 처리를 시작합니다...")
        results = assistant.process_audio_files()
        
        for filename, text, summary, predicted_class in results:
            print("\n" + "="*50)
            print(f"파일명: {filename}")
            print(f"텍스트 요약: {summary}")
            print(f"응급도 분류: {predicted_class}")
            print("="*50 + "\n")
            
            if predicted_class <= 2:  # 응급환자
                print("⚠️ 응급환자로 분류되었습니다!")
                
                # 주소 입력 및 예외 처리
                while True:
                    try:
                        address = input("\n🏥 현재 위치 주소를 입력해주세요: ")
                        recommendations = assistant.handle_emergency(address)
                        break
                    except ValueError as e:
                        print(f"\n❌ 오류: {e}")
                        print("주소를 다시 입력해주세요.")
                    except Exception as e:
                        print(f"\n❌ 예상치 못한 오류가 발생했습니다: {e}")
                        print("주소를 다시 입력해주세요.")
                
                # 추천 병원 정보 출력
                print("\n📍 추천 병원 목록:")
                for idx, row in recommendations.iterrows():
                    print("\n" + "-"*50)
                    print(f"🏥 병원 이름    : {row['name']}")
                    print(f"📍 주소        : {row['address']}")
                    print(f"🚗 이동 거리    : {row['distance']:.0f}m")
                    print(f"⏱️ 예상 소요시간 : {row['duration']:.0f}분")
                    print("-"*50)
                
                # 지도 시각화
                try:
                    print("\n🗺️ 지도를 생성하는 중입니다...")
                    lat, lng = assistant.get_lat_long(address)
                    user_coords = (lat, lng)
                    hospital_map = assistant.visualize_hospitals_with_route(user_coords, recommendations)
                    
                    # 지도 저장
                    map_filename = "hospital_routes.html"
                    hospital_map.save(map_filename)
                    print(f"\n✅ 지도가 '{map_filename}'로 저장되었습니다.")
                    
                    # 주피터 노트북 환경에서 지도 표시
                    try:
                        from IPython.display import display
                        display(hospital_map)
                    except ImportError:
                        print("\n📝 지도 파일을 웹 브라우저에서 열어주세요.")
                    
                except Exception as e:
                    print(f"\n❌ 지도 생성 중 오류가 발생했습니다: {e}")
            
            else:
                print("ℹ️ 응급환자가 아닌 것으로 분류되었습니다.")
                print("일반 진료를 권장드립니다.")
            
            print("\n다음 음성 파일 처리를 시작합니다...\n")
    
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류가 발생했습니다: {e}")
    
    finally:
        print("\n프로그램을 종료합니다.")
