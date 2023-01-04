from sklearn.feature_extraction.text import TfidfVectorizer

from database import diseases
from sklearn.metrics.pairwise import cosine_similarity



def preprocess(symtoms):
    sym_text = ''
    for symtom in symtoms:
        sym_text = sym_text + ' ' + symtom.replace(' ', '_')
    return sym_text

train_data = []
for disease in diseases:
    advance_symtoms = disease['advance_symtoms']
    base_symtoms = disease['base_symtoms']
    train_data.append(preprocess(advance_symtoms))

vectorizer = TfidfVectorizer(min_df=0)
vectorizer.fit(train_data)


def get_vector(symtoms):
    ft = preprocess(symtoms)
    vec = vectorizer.transform([ft])
    return vec

def infer_tfidf(symtoms):
    macthed = []
    for disease in diseases:
        

        disease_symtom = disease['base_symtoms'] + disease['advance_symtoms']
        user_symtom = list(set(symtoms) & set(disease_symtom))

        user_symtom_vec = get_vector(user_symtom)
        disease_vec = get_vector(disease_symtom)

        score = cosine_similarity(user_symtom_vec, disease_vec)[0][0]

        if score > 0:
            macthed.append((disease['name'], score))
    return macthed

# print(infer_tfidf(['sốt cao', 'mũi nghẹt', 'hốc miệng có mảng trắng','da có mụn nước','đau đầu']))