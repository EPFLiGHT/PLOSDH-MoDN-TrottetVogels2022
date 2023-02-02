# script to produce the plot of the clustering of the state with overlayed
# colors for the different diseases
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from colorutils import Color
from sklearn.manifold import TSNE  # T-Distributed Stochastic Neighbor Embedding

from utils import preprocess_data_epoct, shuffle


# functions
def compute_state(qst, encoders, X_cont, X_cat, set="train"):
    """compute the state"""
    if set == "train":
        group_order_features = qst.train_group_order_features
    elif set == "valid":
        group_order_features = qst.valid_group_order_features
    else:
        group_order_features = qst.test_group_order_features
    state = torch.tile(initial_state.state_value, (len(X_cont), 1))

    # apply trained encoders for each level in the tree
    for level in group_order_features.keys():
        for feature_group in group_order_features[level].keys():
            patients = group_order_features[level][feature_group]
            for feature_name in shuffle(feature_group):
                if len(patients) > 0:
                    if feature_name in qst.continuous_features_names:
                        state[patients, :] = encoders[feature_name](
                            state[patients, :],
                            X_cont[
                                patients,
                                qst.continuous_features_names.index(feature_name),
                            ].view(-1, 1),
                        )
                    else:
                        state[patients, :] = encoders[feature_name](
                            state[patients, :],
                            X_cat[
                                patients,
                                qst.categorical_features_names.index(feature_name),
                            ].view(-1, 1),
                        )
    return state


def plot_scatter_cluster(qst, count_dict, diseases_to_keep, decomposition):
    """overlap tsne of state with disease color"""

    def sum_colors(list_colors):
        if len(list_colors) == 1:
            return list_colors[0].hex
        else:
            color = list_colors[0]
            for elem in range(1, len(list_colors)):
                color += list_colors[elem]
            return color.hex

    mapping = {
        "dxfinal_anemia": ("Anemia", Color((50, 50, 200))),
        "dxfinal_dehyd": ("Dehydration", Color((50, 50, 50))),
        "dxfinal_diarrhea": ("Diarrhea", Color((255, 140, 0))),
        "dxfinal_fws": ("FWS", Color((0, 150, 0))),
        "dxfinal_malaria": ("Malaria", Color((150, 75, 0))),
        "dxfinal_malnut": ("Malnutrition", Color((50, 50, 50))),
        "dxfinal_pna": ("Pneumonia", Color((170, 0, 0))),
        "dxfinal_urti": ("URTI", Color((47, 79, 79))),
    }

    clusters = OrderedDict([(index, ["Other"]) for index in range(len(decomposition))])
    colors = OrderedDict(
        [(index, [Color((169, 169, 169))]) for index in range(len(decomposition))]
    )
    for elem in diseases_to_keep:

        if (
            len(
                [
                    qst.disease_names[index]
                    for index, value in enumerate(elem)
                    if value == 1
                ]
            )
            > 0
        ):
            for key in count_dict[elem]:
                disease_list = [
                    mapping[qst.disease_names[index]][0]
                    for index, value in enumerate(elem)
                    if value == 1
                ]
                disease_list.reverse()
                clusters[key] = disease_list
                colors[key] = [
                    mapping[qst.disease_names[index]][1]
                    for index, value in enumerate(elem)
                    if value == 1
                ]

        else:
            for key in count_dict[elem]:
                clusters[key] = ["Other"]
    clusters = [value for key, value in clusters.items()]
    colors = [value for key, value in colors.items()]
    colors = [sum_colors(value) for value in colors]
    clusters = ["<br>+ ".join(elem) for elem in clusters]

    color_mapping = {}
    for index, elem in enumerate(clusters):
        if elem not in color_mapping.keys():
            color_mapping[elem] = colors[index]

    plot_df = pd.DataFrame(
        {
            "Diagnosis": clusters,
            "t-SNE first dimension": decomposition[:, 0],
            "t-SNE second dimension": decomposition[:, 1],
        }
    ).sort_values(by=["Diagnosis"])
    fig = px.scatter(
        plot_df,
        x="t-SNE first dimension",
        y="t-SNE second dimension",
        color="Diagnosis",
        color_discrete_map=color_mapping,
        opacity=0.8,
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig.update_traces(marker={"size": 14})
    fig.update_layout(legend_tracegroupgap=20)
    fig.update_layout(
        font=dict(
            size=18,
        )
    )
    # to plot
    fig.write_image("models/saved_plots/state_tsne.pdf", width=1500, height=900)
    fig.show()
    return


# reproducibility
seed = 0
random.seed(seed)
# load qst_obj
qst_obj = torch.load("models/saved_objects/qst_epoct.pt")
# load pretrained model
model_dict = torch.load(
    "models/saved_objects/model.pt", map_location=torch.device("cpu")
)
encoders = model_dict["encoders"]
single_disease_decoders = model_dict["dis_decoders"]
initial_state = model_dict["initial_state"]

subset = "train"
with torch.no_grad():
    (
        X_train,
        X_valid,
        X_test,
        X_cont_train,
        X_cont_valid,
        X_cont_test,
        X_cat_train,
        X_cat_valid,
        X_cat_test,
        y_train,
        y_valid,
        y_test,
        mean_cont,
        std_cont,
        min_cont,
        max_cont,
    ) = preprocess_data_epoct(
        qst_obj, valid_size=0.2, test_size=0.2, tensors=True, imput_strategy=None
    )
    if subset == "train":
        X_cont = X_cont_train
        X_cat = X_cat_train
        y = y_train
    elif subset == "test":
        X_cont = X_cont_test
        X_cat = X_cat_test
        y = y_test
    state = compute_state(qst_obj, encoders, X_cont, X_cat, set=subset)

    count_dict = {tuple(elem): [] for elem in np.unique(y, axis=0)}
    for elem in np.unique(y, axis=0):
        for patient in range(len(y)):
            if (y[patient, :] == torch.tensor(elem)).all():
                count_dict[tuple(elem)].append(patient)

    # dropping diarrhea
    disease_combi_to_keep = [
        key for key, value in count_dict.items() if len(value) > 70 and key[2] != 1
    ]
    tsne_2d = TSNE(n_components=2, perplexity=10, random_state=seed).fit_transform(
        state
    )
    plot_scatter_cluster(qst_obj, count_dict, disease_combi_to_keep, tsne_2d)


print("End of script")
