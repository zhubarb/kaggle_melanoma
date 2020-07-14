from fastai import *
from fastai.tabular import *
from fastai.vision import *

PATH = '/media/Datas/ML-Data/melanoma'

# distinguish categorical and continuous variables, and dependent variable
cat_names = ['sex', 'anatom_site_general_challenge']
cont_names = ['age_approx']
dep_var = 'benign_malignant'

# transformations for image augmentation
tfms = get_transforms(do_flip=True,
                      flip_vert=True,
                      max_rotate=15,
                      max_zoom=1.05,
                      max_warp=0,
                      max_lighting=0,
                      )


def get_dataframe():
    """Returns DataFrame containing tabular data, image names, and targets."""

    # main data set
    df = pd.read_csv(os.path.join(PATH, 'train.csv'))

    # isolate useful columns
    df = df[cont_names + cat_names + ['image_name'] + [dep_var]]

    return df


def get_val_idxs(n, seed=1234):
    np.random.seed(seed)
    return np.random.permutation(np.arange(n))[0:int(0.2 * n)]


class TabConvDataset(Dataset):
    """A Dataset of combined tabular data, image names, and targets."""

    def __init__(self, x_tab, x_img, y):
        self.x_tab, self.x_img, self.y = x_tab, x_img, y

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return (self.x_tab[i], self.x_img[i]), self.y[i]


class TabConvModel(nn.Module):
    """A combined neural network using the convnet and tabular model"""

    def __init__(self, tab_model, img_model, layers, drops):
        super().__init__()
        self.tab_model = tab_model
        self.img_model = img_model
        lst_layers = []

        activs = [nn.ReLU(inplace=True), ] * (len(layers) - 2) + [None]

        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activs):
            lst_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)

        self.layers = nn.Sequential(*lst_layers)

    def forward(self, *x):
        x_tab = self.tab_model(*x[0])
        x_img = self.img_model(x[1])

        x = torch.cat([x_tab, x_img], dim=1)
        return self.layers(x)


def initialize_combined_model(n_lin_tab=16, n_lin_conv=32, ps_final=0.25, bs=64, sz=502,
                              seed=1234, val_pct=0.2, img_tfms=tfms):
    """Initialize a combined model that can learn from both tabular and image data.
    Params
    ------
    n_lin_tab: int
        number of linear nodes in the single hidden layer for the tabular model
        (default is )
    n_lin_conv: int
        number of linear nodes in the final dense layer for the convolutional model
        (default is 32)
    ps_final: float
        dropout fraction in the final linear layers (default is 0.25)
    bs: int
        batchsize for loading combined TabConvData (default is 64)
    sz: int
        image size
    seed: int, optional
        random seed passed to numpy (default is 1234)
    val_pct: float, optional
        fraction of data used for validation (default is 0.2)
    img_tfms: Transforms, optional
        set of transformations for image augmentation (global default set)
    Returns
    -------
    learn: Learner
        combined `Learner` object built on top of the TabConvModel and
        supplied data
    """

    df = get_dataframe()
    val_idxs = get_val_idxs(len(df))

    # preprocessing
    procs = [FillMissing, Categorify, Normalize]

    # set up tabular data and learner
    tab_data = (TabularList.from_df(df, path=PATH, cat_names=cat_names, cont_names=cont_names, procs=procs)
                .split_by_idx(valid_idx=val_idxs)
                .label_from_df(cols=dep_var, label_cls=CategoryList)
                .databunch(bs=bs))

    tab_learn = tabular_learner(tab_data, layers=[n_lin_tab])

    # set up image data and learner
    img_data = (ImageList.from_df(path=PATH,
                                  df=df,
                                  folder='jpeg/train',
                                  cols='image_name',
                                  suffix='.jpg')
                .split_by_idx(valid_idx=val_idxs)
                .label_from_df(cols=dep_var, label_cls=CategoryList)
                .transform(img_tfms, size=sz)
                .databunch(bs=bs)
                )

    img_learn = cnn_learner(img_data, models.resnet34,
                            pretrained=True,
                            )

    # combined data
    train_ds = TabConvDataset(tab_data.train_ds.x, img_data.train_ds.x, tab_data.train_ds.y)
    valid_ds = TabConvDataset(tab_data.valid_ds.x, img_data.valid_ds.x, tab_data.valid_ds.y)

    train_dl = DataLoader(train_ds, bs)
    valid_dl = DataLoader(valid_ds, 2 * bs)

    data = DataBunch(train_dl, valid_dl, path=PATH)

    # chop off final layers from both models
    tab_learn.model.layers = tab_learn.model.layers[:-2]
    img_learn.model[-1] = nn.Sequential(*img_learn.model[-1][:-5], nn.Linear(1024, n_lin_conv, bias=True),
                                        nn.ReLU(inplace=True))

    lin_layers = [n_lin_tab + n_lin_conv, 2]
    ps = [ps_final]

    # initialize model
    model = TabConvModel(tab_learn.model, img_learn.model, lin_layers, ps)

    layer_groups = [nn.Sequential(*flatten_model(img_learn.layer_groups[0])),
                    nn.Sequential(*flatten_model(img_learn.layer_groups[1])),
                    nn.Sequential(*(flatten_model(img_learn.layer_groups[2]) +
                                    flatten_model(model.tab_model) +
                                    flatten_model(model.layers)))
                    ]

    # combined learner
    learn = Learner(data, model,
                    layer_groups=layer_groups,
                    )

    return learn


if __name__ == '__main__':
    learn = initialize_combined_model()

    learn.freeze()
    learn.fit_one_cycle(1, 1e-2)
    learn.save('combined-init-train-1')

    learn.freeze_to(-2)
    learn.fit_one_cycle(1, 1e-2)
    learn.save('combined-init-train-2')

    learn.unfreeze()
    learn.fit_one_cycle(20, 1e-3)
    learn.save('combined-init-train-3')
