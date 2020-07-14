from fastai import *
from fastai.tabular import *
from fastai.vision import *
from sklearn import model_selection


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


def initialize_combined_model(path, tab_data_dict, img_data_dict, bs=64, val_pct=0.2):
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

    # assert the overlapping image and data info are consistent
    assert img_data_dict['dep_var'] == tab_data_dict[
        'dep_var'], 'Image and tabular model dependent variables do not match'

    train_df = pd.read_csv(os.path.join(path, 'train.csv'))
    tab_train_df = train_df[tab_data_dict['cont_names']+tab_data_dict['cat_names']+[tab_data_dict['dep_var']]]
    img_train_df=train_df[['image_name', 'benign_malignant']] # just need the image names and the output var
    test_df = pd.read_csv(os.path.join(path, 'test.csv'))

    train_idxs, val_idxs = get_stratified_val_idx(train_df, tab_data_dict['dep_var'],
                                      val_pct)

    # set up tabular data and learner
    tab_data = (TabularList.from_df(tab_train_df, path=path, cat_names=tab_data_dict['cat_names'],
                                    cont_names=tab_data_dict['cont_names'],
                                    procs=tab_data_dict['procs'])
                                    .split_by_idx(valid_idx=val_idxs)
                                    .label_from_df(cols=tab_data_dict['dep_var'],
                                                   label_cls=tab_data_dict['label_cls'])
                                    .databunch(bs=bs))

    # tab_learn = tabular_learner(tab_data,
    #                             layers=[100, 50], ps=[0.001, 0.01],
    #                             emb_drop=0.04) # infer loss function automatically loss_func=loss_function,
    tab_learn = tabular_learner(tab_data, layers=[16])

    # set up image data and learner transformations for image augmentation
    img_tfms = get_transforms()

    img_data = (ImageList.from_df(path=path,
                                  df=img_train_df,
                                  cols='image_name',
                                  folder=img_data_dict['folder'],
                                  suffix='.jpg')
                .split_by_idx(valid_idx=val_idxs)
                .label_from_df(cols=img_data_dict['dep_var'], label_cls=img_data_dict['label_cls'])
                .transform(img_tfms, size=img_data_dict['img_size'])
                .databunch(bs=bs)#.normalize(imagenet_stats) TO DO
                )

    img_learn = cnn_learner(img_data, models.resnet34,
                            pretrained=True
                            ) #  nfer loss function automatically  loss_func=loss_function,

    # combined data
    train_ds = TabConvDataset(tab_data.train_ds.x, img_data.train_ds.x, tab_data.train_ds.y)
    valid_ds = TabConvDataset(tab_data.valid_ds.x, img_data.valid_ds.x, tab_data.valid_ds.y)

    train_dl = DataLoader(train_ds, bs)
    valid_dl = DataLoader(valid_ds, 2 * bs)

    combined_data = DataBunch(train_dl, valid_dl, path=path)

    # chop off final layers from both models
    n_lin_conv = 32 # number of linear nodes in the final dense layer for the convolutional model
    n_lin_tab = 16 # number of linear nodes in the last hidden layer for the tabular model
    ps_final = 0.05 # dropout fraction in the final linear layers (default is 0.25)
    assert tab_learn.model.layers[-1].out_features == img_learn.model[-1][-1].out_features, 'Output layer sizes dont match'
    # if standard regression n_output will be 1, and binary classification 2,
    # however this info should be taken from either len(data.classes), or more reliably from the
    # output layer out_sizes from the constituent tab adn conv nets.
    n_output =tab_learn.model.layers[-1].out_features
    # get rid of the output layer and the dropout layer before outputlayer
    tab_learn.model.layers = tab_learn.model.layers[:-2]
    type(tab_learn.model) # fastai.tabular.models.TabularModel (standard torch)
    # replace the whole output block model[-1] of the CNN (not just the final output layer)
    img_learn.model[-1] = nn.Sequential(*img_learn.model[-1][:-5], nn.Linear(1024, n_lin_conv, bias=True),
                                        nn.ReLU(inplace=True))
    type(img_learn.model) # torch.nn.modules.container.Sequential (modules containers, with multiple seq models stacked)

    lin_layers = [n_lin_tab + n_lin_conv, n_output]
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
    learn = Learner(combined_data, model,
                    layer_groups=layer_groups,
                    )

    return learn

def get_stratified_val_idx(df, dep_var, val_percent):

    kf = model_selection.StratifiedKFold(n_splits=int(1/val_percent), shuffle=True, random_state=42)
    fold_idxs=list(kf.split(X=df, y=df[dep_var].values))
    val_idx =  fold_idxs[0][1] # the val idx from the 0th fold
    train_idx = fold_idxs[0][0] # the val idx from the 0th fold

    return train_idx, val_idx

