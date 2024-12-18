import requests
import json
import re
from collections import Counter

class BingSearchProcessor:
    def __init__(self, subscription_key, search_mode=False):
        self.subscription_key = subscription_key
        self.search_mode = search_mode  # 검색 모드 플래그 추가
        self.visual_search_url = "https://api.bing.microsoft.com/v7.0/images/visualsearch"
        self.entity_search_url = "https://api.bing.microsoft.com/v7.0/entities"
        self.news_search_url = "https://api.bing.microsoft.com/v7.0/news/search"
        self.web_search_url = "https://api.bing.microsoft.com/v7.0/search"
        self.headers = {'Ocp-Apim-Subscription-Key': self.subscription_key}

    def set_search_mode(self, mode):
        """검색 모드 설정"""
        self.search_mode = mode

    def get_search_mode(self):
        """현재 검색 모드 상태 반환"""
        return self.search_mode
    
    def get_entity_description(self, object_name):
        """Entity Search API를 사용해 객체 설명 가져오기"""
        params = {"mkt": "en-US", "q": object_name}
        response = requests.get(self.entity_search_url, headers=self.headers, params=params)
        response.raise_for_status()
        entities = response.json().get("entities", {}).get("value", [])

        if not entities:
            return "No description available."

        entity = entities[0]
        hint = entity.get("entityPresentationInfo", {}).get("entityTypeDisplayHint", "")
        description = entity.get("description", "").split(".")[:2]
        return f"{hint}\n- {description[0].strip()}.\n- {description[1].strip()}."

    def identify_object(self, image_path):
        """Visual Search API를 사용해 객체 인식"""
        if not self.search_mode:
            return None

        def extract_names_by_action_type(response_json):
            action_type_names = {"PagesIncluding": [], "VisualSearch": []}
            for tag in response_json.get("tags", []):
                for action in tag.get("actions", []):
                    action_type = action.get("actionType")
                    for item in action.get("data", {}).get("value", []):
                        name = item.get("name")
                        if name:
                            action_type_names[action_type].append(name)
            return action_type_names

        def process_text(names):
            def generate_ngrams(text, n):
                words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
                return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

            combined_text = ' '.join(names)
            return (
                Counter(generate_ngrams(combined_text, 1)),
                Counter(generate_ngrams(combined_text, 2)),
                Counter(generate_ngrams(combined_text, 3)),
            )

        def clean_repetitive_word(result):
            words = result.split()
            return words[0] if len(set(words)) == 1 else result

        def select_final_value(word_counts, bigram_counts, trigram_counts):
            most_common_word, word_freq = word_counts.most_common(1)[0]
            most_common_bigram, bigram_freq = bigram_counts.most_common(1)[0]
            most_common_trigram, trigram_freq = trigram_counts.most_common(1)[0]

            if trigram_freq <= bigram_freq / 2:
                return clean_repetitive_word(most_common_bigram)
            if most_common_word in most_common_trigram:
                return clean_repetitive_word(most_common_trigram)
            return clean_repetitive_word(most_common_bigram)

        with open(image_path, "rb") as image_file:
            response = requests.post(self.visual_search_url, headers=self.headers, files={"image": image_file})
            response.raise_for_status()
            names = extract_names_by_action_type(response.json())
            all_names = names["PagesIncluding"] + names["VisualSearch"]

            if not all_names:
                return "Unknown"

            word_counts, bigram_counts, trigram_counts = process_text(all_names)
            return select_final_value(word_counts, bigram_counts, trigram_counts)

    def get_search_results(self, image_path):
        """통합된 검색 결과 반환"""
        if not self.search_mode:
            return None

        identified_object = self.identify_object(image_path)
        if identified_object == "Unknown":
            return {
                "identified_object": "Unknown object",
                "news": [],
                "web_results": []
            }

        # 뉴스 2개와 웹 검색 결과 2개만 가져오기
        news = self.get_recent_news(identified_object, count=2)
        web_results = self.get_web_search_results(identified_object, count=2)

        return {
            "identified_object": f"Identified Object: {identified_object}",
            "news": news[:2],  # 최대 2개의 뉴스
            "web_results": web_results[:2]  # 최대 2개의 웹 검색 결과
        }

    def get_recent_news(self, object_name, count=2):
        """News Search API를 사용해 최근 뉴스 가져오기"""
        params = {"q": object_name, "mkt": "en-US", "count": count}
        response = requests.get(self.news_search_url, headers=self.headers, params=params)
        response.raise_for_status()
        return [
            f"- {article['name']}"
            for article in response.json().get("value", [])
        ] or ["No recent news found."]

    def get_web_search_results(self, object_name, count=2):
        """Web Search API를 사용해 웹 결과 가져오기"""
        params = {"q": object_name, "mkt": "en-US", "count": count}
        response = requests.get(self.web_search_url, headers=self.headers, params=params)
        response.raise_for_status()
        return [
            f"- {result['name']}"
            for result in response.json().get("webPages", {}).get("value", [])
        ] or ["No web results found."]

if __name__ == "__main__":
    # 테스트 실행
    API_KEY = ""
    processor = BingSearchProcessor(API_KEY)

    IMAGE_PATH = "result/obj_3_sr.jpg"
    identified_object = processor.identify_object(IMAGE_PATH)
    description = processor.get_entity_description(identified_object)
    recent_news = processor.get_recent_news(identified_object)
    web_results = processor.get_web_search_results(identified_object)

    print(f"Identified Object: {identified_object}")
    print(f"Description: {description}")
    print("Recent News:")
    print("\n".join(recent_news))
    print("Web Search:")
    print("\n".join(web_results))
