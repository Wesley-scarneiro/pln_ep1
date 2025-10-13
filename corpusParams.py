class CorpusParam:
    def __init__(self,
            path: str,
            clean: bool,
            sublinear: bool,
            min_df: int,
            ngram_range: tuple,
            max_features: int,
            logreg_c: float,
            penalty: str,
            solver: str
            ):
        self.__name = path.split('/')[1]
        self.__path = path
        self.__clean = clean
        self.__sublinear = sublinear
        self.__min_df = min_df
        self.__ngram_range = ngram_range
        self.__max_features = max_features
        self.__logreg_C = logreg_c
        self.__penalty = penalty
        self.__solver = solver

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
    def sublinear(self) -> bool:
        return self.__sublinear
    
    @property
    def min_df(self) -> int:
        return self.__min_df
    
    @property
    def ngram_range(self) -> tuple[int, int]:
        return self.__ngram_range
    
    @property
    def max_features(self) -> int:
        return self.__max_features
    
    @property
    def penalty(self) -> str:
        return self.__penalty
    
    @property
    def logreg_C(self) -> float:
        return self.__logreg_C
    
    @property
    def solver(self) -> str:
        return self.__solver