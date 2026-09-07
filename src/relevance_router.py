class RelevanceRouter:

    def __init__(
        self,
        threshold=0.20
    ):

        self.threshold = threshold

    def is_relevant(
        self,
        scores
    ):

        if scores is None:
            return False

        if len(scores) == 0:
            return False

        best_score = max(scores)

        return best_score >= self.threshold

    def get_best_score(
        self,
        scores
    ):

        if not scores:
            return None

        return float(
            max(scores)
        )