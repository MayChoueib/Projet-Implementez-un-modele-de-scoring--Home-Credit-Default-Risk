# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request
import pickle
import json
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import shap

# load data

df_test = pd.read_csv("sample_test_all.csv", index_col='SK_ID_CURR')
customers_ids = list(df_test.index.to_series())
df_infos = pd.read_csv("sample_test_infos.csv", index_col='SK_ID_CURR')
df_feat_imp = pd.read_csv("feature_importances_test")


# load the model
# Light GBM model avec balanced_class
best_model_bank = pickle.load(open('lgb_bank_tot', 'rb'))
# Explainer shap pour l'interprétabilité locale
explainer = shap.TreeExplainer(best_model_bank)
# Seuil optimal du modèle pour classer les clients en défaillants ou non
best_treshold = 0.1


def neigb_mod(df):
    categ = df['class']
    df_num = df.select_dtypes('number').drop(['score', 'class'], axis=1)
    scaler = MinMaxScaler().fit(df_num)
    df_num_norm = pd.DataFrame(scaler.transform(df_num), columns=df_num.columns,
                               index=df.index)
    neigh = NearestNeighbors(n_neighbors=20)
    neighbors = neigh.fit(df_num_norm)
    df_num_norm['class'] = categ
    return neighbors, df_num_norm


neighbors, df_infos_norm = neigb_mod(df_infos)
infos_json = json.loads(df_infos.to_json())
infos_norm_json = json.loads(df_infos_norm.to_json())


app = Flask(__name__)


# retourner ids clients 'SK_ID_CURR'
@app.route('/cust_ids/')
def clients_ids():
    # Return en json format
    return jsonify({'status': 'ok',
                    'data': customers_ids})


# Retourner les infos descriptives d'un client (SK_ID_CURR)
@app.route('/data_client/', methods=['GET', 'POST'])
def data_client():
    # Parse the http request to get arguments ('SK_ID_CURR')
    id_client = int(request.args.get('id_client'))
    # infos descriptives pour le client (id_client)
    feat_client = df_infos.loc[id_client].drop(['score', 'class', 'decision'])

    # Return data
    return jsonify({'data': json.loads(feat_client.to_json())})


# Retourner les infos descriptives de tous les clients
@app.route('/infos_desc_all_clients/')
def all_data_clients():
    return jsonify({'scores_clients': infos_json,
                    'norm_scores_clients': infos_norm_json})


# Retourner feature importance global
@app.route('/feat_imp_global/')
# get globals features importance
def feat_imp_glob():
    df_feat_imp_js = json.loads(df_feat_imp.to_json())
    return jsonify({'feat_imp_global': df_feat_imp_js})


# Retourner le score et la decision pour un client
# si defaillant (credit refused) sinon (credit granted)
@app.route('/score/', methods=['GET', 'POST'])
def scoring():
    # Parse the http request to get arguments ('SK_ID_CURR')
    id_client = int(request.args.get('id_client'))
    # data personnel pour le client (id_client)
    feat_client = df_test.loc[id_client]
    # prediction score pour le client
    score_client = round(best_model_bank.predict_proba([feat_client])[0][1], 3)
    if score_client >= best_treshold:
        select = "Refused "
    else:
        select = "Granted"
    return jsonify({'id_client': id_client,
                    'score': str(score_client),
                    'decision': select})


# Infos descriptives des plus proches voisins
def get_df_voisins(id_client):
    feat_client_norm = df_infos_norm.loc[id_client].drop('class').to_numpy().reshape(1, -1)
    idx = neighbors.kneighbors(feat_client_norm, return_distance=False)
    df_voisins = df_infos.iloc[idx[0], :].select_dtypes('number')
    df_voisins_norm = df_infos_norm.iloc[idx[0], :]
    return df_voisins, df_voisins_norm



@app.route("/clients_similaires/", methods=["GET"])
def data_voisins():
    id_client = int(request.args.get("id_client"))
    df_voisins, df_voisins_norm = get_df_voisins(id_client)
    #df_client_norm_js = json.loads(df_client_norm.to_json())
    df_voisins_jsn = json.loads(df_voisins.to_json())  # .to_json(orient='index')
    df_voisins_norm_jsn = json.loads(df_voisins_norm.to_json())
    return jsonify({'df_voisins': df_voisins_jsn,
                    'df_voisins_norm': df_voisins_norm_jsn})


# Get locales features importance  du client shap values
@app.route("/feat_imp/", methods=["GET"])
def shap_clients():
    id_client = int(request.args.get("id_client"))
    feat_client = df_test.loc[id_client]
    df_feat_client = pd.DataFrame(df_test.loc[id_client]).T
    df_feat_client = df_feat_client.apply(pd.to_numeric)
    df_name_features = df_feat_client.columns
    sv = explainer(df_feat_client)
    # Return data
    return jsonify({'status': 'ok',
                    'shap_values': sv.values[0, :, 1].tolist(),
                    'base_value': (sv.base_values[0, 1]),
                    'data': feat_client.values.tolist(),
                    'feature_names': df_name_features.tolist()})





if __name__ == "__main__":
    app.run(port=5555, debug=True)
