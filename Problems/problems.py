import numpy as np

from abc import ABCMeta, abstractmethod
class ObjectiveFunction(metaclass=ABCMeta):
    def __init__(self, nvar):
        self.nvar = nvar
        self.xmin = np.empty(nvar)
        self.xmax = np.empty(nvar)
        self.set_xmin()
        self.set_xmax()
        self.penalty_factor = 1
        self.tolerance_factor = None
        
    @abstractmethod
    def evaluate(self, x):
        pass

    @abstractmethod
    def set_xmin(self):
        pass
    
    @abstractmethod
    def set_xmax(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def set_penalty_factors(self, penalty_factor, tolerance_factor, penalty_exp):
        self.penalty_factor = penalty_factor
        self.tolerance_factor = tolerance_factor
        self.penalty_exp = penalty_exp

    def get_nvar(self):
        return self.nvar

    def get_xmin(self):
        return self.xmin

    def get_xmin_at(self, index):
        return self.xmin[index]
    
    def get_xmax(self):
        return self.xmax
    
    def get_xmax_at(self, index):
        return self.xmax[index]

    def evaluate_penalty(self,x):
        if hasattr(self, 'constraint_penalty') and callable(getattr(self, 'constraint_penalty')):
            violations, num_violations = self.constraint_penalty(x)

            not_weighted_penalty = sum(violations) 
            weighted_penalty = self.penalty_factor * sum((1+violation)**self.penalty_exp for violation in violations) # squared
            return weighted_penalty, not_weighted_penalty, num_violations
        else:
            return (0,0,0)

class sphere(ObjectiveFunction):        
    def evaluate(self, x):
        result = 0.0
        for i in range(self.nvar):
            result = result + x[i] ** 2
        return result
    
    def set_xmin(self):
        for i in range(self.nvar):
            self.xmin[i] = -5.0

    def set_xmax(self):
        for i in range(self.nvar):
            self.xmax[i] = 5.0
    
    def get_name(self):
        return sphere.__name__
   

class Layeb05(ObjectiveFunction):        
    def evaluate(self, x):
        result = 0.0
        for i in range(self.nvar - 1):
            A = np.sin((x[i]- np.pi/2)) + np.cos(x[i+1]- np.pi)
            B = np.cos(2*x[i]-x[i+1]+np.pi/2)
            result = result + np.log(np.abs(A) + 0.001 )/ (np.abs(B) + 1)
        return result
    
    def set_xmin(self):
        for i in range(self.nvar):
            self.xmin[i] = -10

    def set_xmax(self):
        for i in range(self.nvar):
            self.xmax[i] = 10
    
    def get_name(self):
        return Layeb05.__name__

class Layeb10(ObjectiveFunction):        
    def evaluate(self, x):
        result = 0.0
        for i in range(self.nvar - 1):
            A = np.log(x[i]**2 + x[i+1]**2 + 0.5)
            B = 100 * np.sin(x[i] + x[i+1])
            result = result + A ** 2 + np.abs(B)
        return result
    
    def set_xmin(self):
        for i in range(self.nvar):
            self.xmin[i] = -10

    def set_xmax(self):
        for i in range(self.nvar):
            self.xmax[i] = 10
    
    def get_name(self):
        return Layeb10.__name__

class Layeb15(ObjectiveFunction):        
    def evaluate(self, x):
        result = 0.0
        for i in range(self.nvar - 1):
            A = np.sqrt(np.tanh(2*np.abs(x[i]) - x[i+1] ** 2 - 1))
            B = np.exp(x[i]*x[i+1]+1)-1
            result = result + 10*A + np.abs(B)
        return result
    
    def set_xmin(self):
        for i in range(self.nvar):
            self.xmin[i] = -100

    def set_xmax(self):
        for i in range(self.nvar):
            self.xmax[i] = 100
    
    def get_name(self):
        return Layeb15.__name__

class Layeb18(ObjectiveFunction):        
    def evaluate(self, x):
        result = 0.0
        for i in range(self.nvar - 1):
            A = np.cos(2*x[i]*x[i+1]/np.pi)
            B = np.sin(x[i]+x[i+1])*np.cos([x[i]])
            result = result + np.log(A + 0.001) / (np.abs(B) + 1)
        return result
    
    def set_xmin(self):
        for i in range(self.nvar):
            self.xmin[i] = -10

    def set_xmax(self):
        for i in range(self.nvar):
            self.xmax[i] = 10
    
    def get_name(self):
        return Layeb18.__name__

class rastrigin(ObjectiveFunction):     
    def evaluate(self, x):
        result = 0.0
        for i in range(self.nvar):
            result = result + x[i]*x[i] - 10*np.cos(2*np.pi*x[i])
        result = result + 10*self.nvar
        return result
    
    def set_xmin(self):
        for i in range(self.nvar):
            self.xmin[i] = -5.12
    
    def set_xmax(self):
        for i in range(self.nvar):
            self.xmax[i] = 5.12
    
    def get_name(self):
        return rastrigin.__name__

class rosenbrock(ObjectiveFunction):       
    def evaluate(self, x):
        result = 0.0
        for i in range(self.nvar - 1):
            result = result + 100*np.power(x[i + 1] - x[i]*x[i], 2) + np.power(1 - x[i], 2)
        return result
    
    def set_xmin(self):
        for i in range(self.nvar):
            self.xmin[i] = -10.0
    
    def set_xmax(self):
        for i in range(self.nvar):
            self.xmax[i] = 10.0
    
    def get_name(self):
        return rosenbrock.__name__

class G1(ObjectiveFunction):
    def __init__(self):
        super().__init__(nvar=13)

    def evaluate(self, x):
        term1 = 5 * (x[0] + x[1] + x[2] + x[3])
        term2 = -5 * sum(np.power(x[i], 2) for i in range(4))
        term3 = -sum(x[i] for i in range(4, 13))

        result = term1 + term2 + term3
        return result
    
    def set_xmin(self):
        for i in range(self.nvar):
            self.xmin[i] = 0.0

    def set_xmax(self):
        for i in range(9):
            self.xmax[i] = 1.0
        for i in range(9, 12):
            self.xmax[i] = 100.0
        self.xmax[12] = 1.0

    def constraint_penalty(self, x):
        """
        Returns a tuple with the total penalty (weighted by the penalty factor), penalty (not weighted by penalty factor), number of violations
        """
        constraints = [
            lambda x: 2 * x[0] + 2 * x[1] + x[9] + x[10] - 10,
            lambda x: 2 * x[0] + 2 * x[2] + x[9] + x[11] - 10,
            lambda x: 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10,
            lambda x: 8 * x[0] + x[9],
            lambda x: -8 * x[1] + x[10],
            lambda x: 8 * x[2] + x[11],
            lambda x: -2 * x[3] - x[4] + x[9],
            lambda x: -2 * x[5] - x[6] + x[10],
            lambda x: -2 * x[7] - x[8] + x[11]
        ]

        violations = [max(0, constraint(x)) for constraint in constraints]
        num_violations = sum(1 for v in violations if v > 0)
        
        return violations, num_violations

    def get_name(self):
        return G1.__name__

class G4(ObjectiveFunction):
    def __init__(self):
        super().__init__(nvar=5)

    def evaluate(self, x):
        result = 5.3578547 * np.power(x[2], 2) + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141

        return result

    def constraint_penalty(self, x):
        """
        Returns a tuple with the total penalty (weighted by the penalty factor), penalty (not weighted by penalty factor), number of violations
        """
        penalty = 0
        
        constraints = [
            lambda x: 85.334407 + 0.0056858 * x[1] * x[4] + 0.00026 * x[0] * x[3] - 0.0022053 * x[2] * x[4] - 92,
            lambda x: - (85.334407 + 0.0056858 * x[1] * x[4] + 0.00026 * x[0] * x[3] - 0.0022053 * x[2] * x[4]),
            lambda x: 90 - (80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * np.power(x[2], 2)),
            lambda x: 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * np.power(x[2], 2)-110,
            lambda x: 20 - (9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3]),
            lambda x: 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3] - 25
        ]
        
        violations = [max(0, constraint(x)) for constraint in constraints]
        num_violations = sum(1 for v in violations if v > 0)
        
        return violations, num_violations

    def set_xmin(self):
        self.xmin = np.array([78.0, 33.0, 27.0, 27.0, 27.0])

    def set_xmax(self):
        self.xmax = np.array([102.0, 45.0, 45.0, 45.0, 45.0])

    def get_name(self):
        return G4.__name__

class G5(ObjectiveFunction):
    def __init__(self):
        super().__init__(nvar=4)

    def evaluate(self, x):
        result = 3 * x[0] + 0.000001 * np.power(x[0], 3) + 2 * x[1] + 0.000002 / 3 * np.power(x[2], 3)

        return result

    def constraint_penalty(self, x):
        """
        Returns a tuple with the total penalty (weighted by the penalty factor and with tolerance), penalty (not weighted by penalty factor
        nor with tolerance), number of violations (without tolerance)
        """
        penalty = 0
        constraints = [
            lambda x: - (x[3] - x[2] + 0.55),
            lambda x: - (x[2] - x[3] + 0.55),
            lambda x: 1000 * np.sin(-x[2] - 0.25) + 1000 * np.sin(-x[3] - 0.25) + 894.8 - x[0] - self.tolerance_factor,
            lambda x: - (1000 * np.sin(-x[2] - 0.25) + 1000 * np.sin(-x[3] - 0.25) + 894.8 - x[0]) + self.tolerance_factor,
            lambda x: 1000 * np.sin(x[2] - 0.25) + 1000 * np.sin(x[3] - 0.25) + 894.8 - x[1] - self.tolerance_factor,
            lambda x: - (1000 * np.sin(x[3] - 0.25) + 1000 * np.sin(x[2] - 0.25) + 1294.8) + self.tolerance_factor
        ]
        
        violations = [max(0, constraint(x)) for constraint in constraints]
        num_violations = sum(1 for v in violations if v > 0)
        
        return violations, num_violations

    def set_xmin(self):
        self.xmin = np.array([0, 0, -0.55, -0.55])

    def set_xmax(self):
        self.xmax = np.array([1200, 1200, 0.55, 0.55])

    def get_name(self):
        return G5.__name__

class G6(ObjectiveFunction):
    def __init__(self):
        super().__init__(nvar=2)

    def evaluate(self, x):
        result = np.power((x[0] - 10), 3) + np.power((x[1] - 20), 3)

        return result 

    def constraint_penalty(self, x):
        """
        Returns a tuple with the total penalty (weighted by the penalty factor), penalty (not weighted by penalty factor), number of violations
        """
        penalty = 0

        constraints = [
            lambda x: -(x[0] - 5) ** 2 - (x[1] - 5) ** 2 + 100,
            lambda x: +(x[0] - 6) ** 2 + (x[1] - 5) ** 2 - 82.81
        ]

        violations = [max(0, constraint(x)) for constraint in constraints]
        num_violations = sum(1 for v in violations if v > 0)
        
        return violations, num_violations

    def set_xmin(self):
        self.xmin = np.array([13, 0])

    def set_xmax(self):
        self.xmax = np.array([100, 100])

    def get_name(self):
        return G6.__name__


class FunctionFactory:
    function_dictionary = {
        sphere.__name__: lambda nvar: sphere(nvar),
        rastrigin.__name__: lambda nvar: rastrigin(nvar),
        rosenbrock.__name__: lambda nvar: rosenbrock(nvar),
        Layeb05.__name__: lambda nvar: Layeb05(nvar),
        Layeb10.__name__: lambda nvar: Layeb10(nvar),
        Layeb15.__name__: lambda nvar: Layeb15(nvar),
        Layeb18.__name__: lambda nvar: Layeb18(nvar),
        G1.__name__: G1,
        G4.__name__: G4,
        G5.__name__: G5,
        G6.__name__: G6
    }

    @classmethod
    def select_function(cls, function_name, nvar=None):
        if function_name in ["G1", "G4", "G5", "G6"]:
            return cls.function_dictionary[function_name]()
        else:
            return cls.function_dictionary[function_name](nvar)