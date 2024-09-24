import numpy as np

class Accuracy:
    """
    A class for calculating various accuracy metrics for binary classification problems,
    particularly useful for land surface model forecasts.
    """

    
    def prep_clf(s, o, threshold=0.1):
        """
        Prepare classification results and calculate confusion matrix elements.

        Args:
            s (numpy.ndarray): Simulated/predicted values
            o (numpy.ndarray): Observed/actual values
            threshold (float): Threshold for binary classification (default: 0.1)

        Returns:
            tuple: (hits, misses, falsealarms, correctnegatives)
        """
        # Binarize the data based on the threshold
        o_binary = o >= threshold
        s_binary = s >= threshold

        # Calculate confusion matrix elements
        hits = np.sum(o_binary & s_binary)
        misses = np.sum(o_binary & ~s_binary)
        falsealarms = np.sum(~o_binary & s_binary)
        correctnegatives = np.sum(~o_binary & ~s_binary)

        return hits, misses, falsealarms, correctnegatives

    
    def bss(s, o, threshold=0.1):
        """
        Calculate Brier Skill Score (BSS).

        Args:
            s (numpy.ndarray): Simulated/predicted values
            o (numpy.ndarray): Observed/actual values
            threshold (float): Threshold for binary classification (default: 0.1)

        Returns:
            float: BSS value
        """
        o_binary = (o >= threshold).astype(float)
        s_binary = (s >= threshold).astype(float)
        return np.sqrt(np.mean((o_binary - s_binary) ** 2))

    
    def hss(s, o, threshold=0.1):
        """
        Calculate Heidke Skill Score (HSS).

        Args:
            s (numpy.ndarray): Simulated/predicted values
            o (numpy.ndarray): Observed/actual values
            threshold (float): Threshold for binary classification (default: 0.1)

        Returns:
            float: HSS value
        """
        hits, misses, falsealarms, correctnegatives = Accuracy.prep_clf(s, o, threshold)
        
        numerator = 2 * (hits * correctnegatives - misses * falsealarms)
        denominator = (misses**2 + falsealarms**2 + 2*hits*correctnegatives +
                       (misses + falsealarms)*(hits + correctnegatives))
        
        return numerator / denominator if denominator != 0 else 0

    
    def bias(s, o, threshold=0.1):
        """
        Calculate Bias Score.

        Args:
            s (numpy.ndarray): Simulated/predicted values
            o (numpy.ndarray): Observed/actual values
            threshold (float): Threshold for binary classification (default: 0.1)

        Returns:
            float: Bias Score
        """
        hits, misses, falsealarms, _ = Accuracy.prep_clf(s, o, threshold)
        return (hits + falsealarms) / (hits + misses) if (hits + misses) != 0 else 0

    # ... (implement other methods similarly)

    
    def fsc(s, o, threshold=0.1):
        """
        Calculate F1 Score.

        Args:
            s (numpy.ndarray): Simulated/predicted values
            o (numpy.ndarray): Observed/actual values
            threshold (float): Threshold for binary classification (default: 0.1)

        Returns:
            float: F1 Score
        """
        hits, misses, falsealarms, _ = Accuracy.prep_clf(s, o, threshold)
        
        precision = hits / (hits + falsealarms) if (hits + falsealarms) != 0 else 0
        recall = hits / (hits + misses) if (hits + misses) != 0 else 0
        
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
