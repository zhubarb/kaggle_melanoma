from tabconvlearner import initialize_combined_model
import os
from fastai.tabular.transform import FillMissing, Categorify, Normalize
from fastai.data_block import CategoryList
import warnings
warnings.filterwarnings('ignore') # # suppress pytorch warnings

if __name__ == '__main__':

    PATH = '/media/Datas/ML-Data/melanoma'

    # distinguish categorical and continuous variables, and dependent variable
    tab_data_dict = {'cat_names': ['sex', 'anatom_site_general_challenge'],
                     'cont_names': ['age_approx'],
                     'dep_var': 'benign_malignant',
                     'procs': [FillMissing, Categorify, Normalize],
                     'label_cls': CategoryList
                     }
    img_data_dict = {'folder': 'jpeg/train',
                     'img_size': 502,
                     'dep_var': 'benign_malignant',
                     'label_cls': CategoryList
                     }
    bs = 32
    print("generating combined tabular & convolutional model")
    learn = initialize_combined_model(path=PATH, tab_data_dict=tab_data_dict, img_data_dict=img_data_dict,
                                      bs=bs, val_pct=0.2)
    print("starting to train combined tabular & convolutional model")
    learn.data.one_batch()
    learn.freeze()
    learn.fit_one_cycle(1, 1e-2)
    learn.save('combined-init-train-1')

    # learn.freeze_to(-2)
    # learn.fit_one_cycle(1, 1e-2)
    # learn.save('combined-init-train-2')
    #
    # learn.unfreeze()
    # learn.fit_one_cycle(20, 1e-3)
    # learn.save('combined-init-train-3')