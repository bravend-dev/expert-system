import pandas as pd

df = pd.read_excel("database.xlsx", sheet_name="disease")
diseases = []

base_symtom_list = set()
for index, row in df.iterrows():
    name = row['name']
    type = row['type']
    base_symtoms = row['base_symtoms'].split('\n')
    for sym in base_symtoms:
        base_symtom_list.add(sym)
    advance_symtoms = row['advance_symtoms'].split('\n')
    diseases.append({
        'name': name, 
        'base_symtoms': base_symtoms,
        'advance_symtoms': advance_symtoms,
        'type': type
    })

base_symtom_list = list(base_symtom_list)