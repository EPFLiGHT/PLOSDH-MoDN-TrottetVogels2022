# mapping dictionnaries of feature codes to names for readability in plots

from unicodedata import name
import torch
import pandas as pd
import os


def get_name_mapping():
    qst_obj = torch.load("saved_objects/qst_epoct.pt")
    # qst_obj = torch.load('models/saved_objects/qst_epoct.pt')
    os.chdir("../data")
    # os.chdir('data')
    feat_dict = pd.read_csv(
        os.path.join(os.getcwd(), "epoct_ezvir_05dec2018_dictionary.csv"),
        encoding="latin1",
    )
    df_labeled = pd.read_csv(
        os.path.join(os.getcwd(), "epoct_ezvir_05dec2018_labeled.csv")
    )
    df_unlabeled = pd.read_csv(
        os.path.join(os.getcwd(), "epoct_ezvir_05dec2018_unlabeled.csv")
    )
    os.chdir("../models")
    df_labeled = df_labeled[qst_obj.feature_names]
    df_labeled["hist_ttt_before"] = (
        df_labeled["hist_ttt_before"].replace("0", "No").replace(0.0, "No")
    )
    df_unlabeled = df_unlabeled[qst_obj.feature_names]
    df = feat_dict[feat_dict.name.isin(qst_obj.feature_names)]
    # clean
    def reduce_text(x):
        if x != x:
            x = "Other complaints"
        else:
            if "URI" in x:
                x = x.replace("URI", "Dysuria")

            for elem in [
                "History: ",
                "D0",
                "Vital Sign: ",
                "Sign: ",
                "(d0)",
                "Symptom: ",
                "Chief Complaint ",
                "DO",
                ",",
                "F23",
                "received ",
                "Severe ",
                "soft ",
                " culture",
                "(mmol/L)",
                ">=2/24h",
                " entered",
                " & age>6 months",
                "Axillary ",
                " status",
                " in months",
                "Any ",
                "initial",
                "Initial",
            ]:
                if elem in x:
                    x = x.replace(elem, "")
            if "pneumococccal" in x:
                x = x.replace("pneumococcal", "pneumoc")
            if "Dusuria" in x:
                x = x.replace("Dusuria", "Dysuria")
            if "RR" in x:
                x = x.replace("RR", "Respiration rate")
            if "Age" in x:
                x = x.replace("Age", "age (in months)")

            x = x.rstrip().lstrip().capitalize()

        return x

    df.loc[:, "varlab"] = df["varlab"].apply(reduce_text)
    # create column with possible dict mapping e.g. sex : {0:'male', 1:'female'}
    # sentence case
    dict_values = {}
    for elem in qst_obj.feature_names:
        if len(df_labeled[elem].unique()) <= 15 and elem != "lab_udip_spec_d0":
            dict_values[elem] = df_labeled[elem].dropna().unique()
    # custom mapping used in preprocessing of features
    dict_mapping = {
        "dem_sex": {"male": 0, "female": 1},
        "lab_bcx_id_d0": {
            "negative": 1,
            "Salmonella paratyphi B": 2,
            "Salmonella paratyphi C": 3,
            "Corynebacterium sp.": 4,
            "Staph non-aureus": 5,
            "Aerococcus  Viridans 2": 6,
            "Photobacterium damselae": 7,
            "Acinetobacter barmanii/coacetius": 8,
            "Micrococcus sp.": 9,
            "Bacillus Spp": 10,
            "Staph aureus": 11,
            "Staph capitis": 12,
            "Vibrio vulnificus ": 13,
            "Shigella spp": 14,
            "Pseudomonas aureginosa": 15,
            "E.coli": 16,
            "Staph chromogenes": 17,
            "Salmonella paratyphi": 18,
            "Staph xylosus": 19,
            "Proteus mirabilis": 20,
            "Providencia alcalifaciens": 21,
            "Pseudomonas oryzihabitans": 22,
            "Stenotrophomonas maltophilia": 23,
            "Salmonella typhi": 24,
            "Streptococcus spp": 25,
        },
        "sign_dehyd_skin_d0": {"normally": 0, "slow": 2},
        "lab_urine_cx_id": {
            "Mixed growth": 1,
            "negative": 2,
            "E. coli": 3,
            "Klebsiella pneumoniae": 4,
            "Raoultella ornithinolytica": 5,
            "Proteus mirabilis": 6,
            "Serratia liquefaciens": 7,
            "E.coli": 8,
            "Staph saprophyticus": 9,
            "Citrobacter koseri/amalon": 10,
            "Staph non-aureus": 11,
            "sample error": 12,
            "Salmonella typhi B": 13,
            "Serratia odorifera": 14,
            "Pasteurella pheumotropica": 15,
        },
        "symp_complaint_o_d0": {
            "No": 0,
            "Boils": 1,
            "Cough": 2,
            "Runny nose": 3,
            "Loose stools": 4,
            "Loss of appetite": 5,
            "Runny nose liss if appetite": 6,
            "Eye discharge": 7,
            "Headache": 8,
        },
        "hist_vac_pcv1": {"No": 0, "Yes": 1, "Unknown": 9},
        "hist_vac_pcv3": {"No": 0, "Yes": 1, "Unknown": 9},
    }

    for key in qst_obj.feature_names:
        if key in ["lab_udip_spec_d0"]:
            continue
        if len(df_unlabeled[key].unique()) <= 21 and key not in dict_mapping.keys():
            dict_mapping[key] = {}
            for value in df_labeled[key].dropna().unique():
                patients = df_labeled[df_labeled[key] == value].index
                mapping = df_unlabeled.loc[patients, key].unique()
                if len(mapping) > 1:
                    print("smth is going wrong")
                else:
                    dict_mapping[key][value] = df_unlabeled.loc[
                        df_labeled[df_labeled[key] == value].index, key
                    ].unique()[0]
    dict_inverse_mapping = {
        key: {value: key_name for key_name, value in dict_mapping[key].items()}
        for key in dict_mapping.keys()
    }
    # very sorry about these lines
    for key in dict_inverse_mapping.keys():
        for key_2, value in dict_inverse_mapping[key].items():
            if isinstance(value, str):
                dict_inverse_mapping[key][key_2] = value.capitalize()
    return df, dict_inverse_mapping


if __name__ == "__main__":
    df, dict_ = get_name_mapping()
    print("End of script")
