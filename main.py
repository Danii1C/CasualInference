import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination, CausalInference
from tabulate import tabulate

# модель Байесовской сети
model = BayesianNetwork([('age', 'smoking'), ('gender', 'smoking'), ('smoking', 'disease'), ('air_pollution', 'disease')])

data = []
with open("generated_data.txt", "r") as file:
    for line in file:
        line = line.strip()
        items = list(map(int, line.split(',')))
        data.append(items)

data = pd.DataFrame(data, columns=['age', 'gender', 'smoking', 'disease', 'air_pollution'])

inference = CausalInference(model)
print(inference.estimate_ate('smoking', 'disease', data=data, estimator_type="linear"))

# Оценка параметров модели на основе данных
model.fit(data)

# Выполнение выводов на основе заданных значений переменных
infer = VariableElimination(model)

query = infer.query(variables=['disease'], evidence={'smoking': 0, 'air_pollution': 2})
# Список значений
state_names_D = model.get_cpds('disease').state_names['disease']
query_table = list(zip(state_names_D, list(query.values)))
headers = ["disease", "Условная вероятность"]
print(tabulate(query_table, headers=headers, tablefmt="grid"))

# is_dependent = model.is_dconnected('age', 'gender',  observed='smoking')
# print(is_dependent)
# print(model.active_trail_nodes(['age', 'gender'], observed='smoking'))


