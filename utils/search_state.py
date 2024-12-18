from threading import Thread, Lock
from utils.bing_search_api import BingSearchProcessor

class SearchState:
    _instance = None
    _bing_api = None
    
    def __new__(cls, bing_api=None):
        if cls._instance is None:
            cls._instance = super(SearchState, cls).__new__(cls)
            cls._instance.current_results = None
            cls._instance.is_processing = False
            cls._instance.lock = Lock()
            cls._bing_api = bing_api
            cls._instance.processor = BingSearchProcessor(cls._bing_api)
        elif bing_api is not None:
            cls._bing_api = bing_api
            cls._instance.processor = BingSearchProcessor(bing_api)
        return cls._instance

    def get_state(self):
        with self.lock:
            return {
                'results': self.current_results,
                'is_processing': self.is_processing
            }
    
    def start_processing(self):
        with self.lock:
            self.is_processing = True
            self.current_results = None
            print("Started processing search")
    
    def finish_processing(self, results):
        with self.lock:
            self.current_results = results
            self.is_processing = False
            print("Search completed successfully")

    def set_search_mode(self, mode):
        self.processor.set_search_mode(mode)
        if not mode:
            with self.lock:
                self.current_results = None
                self.is_processing = False

def search_async(image_path, search_state_instance):  # search_state_instance 파라미터 추가
    def _search():
        try:
            print(f"Starting search for {image_path}")
            search_state_instance.start_processing()
            results = search_state_instance.processor.get_search_results(image_path)
            print("Search completed, updating state...")
            search_state_instance.finish_processing(results)
        except Exception as e:
            print(f"Search error: {e}")
            import traceback
            print(traceback.format_exc())
            search_state_instance.finish_processing(None)
    
    thread = Thread(target=_search)
    thread.daemon = True
    thread.start()
    return thread