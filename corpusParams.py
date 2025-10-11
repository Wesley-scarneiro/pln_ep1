class CorpusParam:
    def __init__(self,
            path: str,
            clean: bool,
            sublinear_tf: bool,
            min_df: int,
            ngram_range: tuple,
            max_features: int
            ):
        self.__name = path.split('/')[1]
        self.__path = path
        self.__clean = clean
        self.__sublinear_tf = sublinear_tf
        self.__min_df = min_df
        self.__ngram_range = ngram_range
        self.__max_features = max_features

    @property
    def name(self) -> str:
        return self.__name

    @property
    def path(self) -> str:
        return self.__path
    
    @property
    def clean(self) -> bool:
        return self.__clean
    
    @property
    def sublinear_tf(self) -> bool:
        return self.__sublinear_tf
    
    @property
    def min_df(self) -> int:
        return self.__min_df
    
    @property
    def ngram_range(self) -> tuple[int, int]:
        return self.__ngram_range
    
    @property
    def max_features(self) -> int:
        return self.__max_features