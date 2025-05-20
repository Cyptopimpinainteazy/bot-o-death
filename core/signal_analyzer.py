class TechnicalAnalysis:
    def __init__(self):
        pass

    def analyze(self, data):
        return {"signal": "buy", "confidence": 0.95}

class SignalAnalyzer:
    def __init__(self, technical_analysis):
        self.technical_analysis = technical_analysis

    def analyze_signals(self, data):
        return self.technical_analysis.analyze(data)
