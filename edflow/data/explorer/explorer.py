import streamlit as st
import importlib
import numpy as np

from edflow.util import walk


from edflow.data.dataset_mixin import DatasetMixin


class TestDset(DatasetMixin):
    def __init__(self, config):
        self.config = config

        self.labels = {'l1': np.arange(100)}

    def get_example(self, idx):
        return {
                'im': np.linspace(0, idx/len(self), num=256**2).reshape([256, 256])[..., None],
                'im2': np.linspace(0, idx/len(self), num=256**2).reshape([256, 256])[..., None]}

    def __len__(self):
        return 100


def process_args():
    from argparse import ArgumentParser

    A = ArgumentParser()
    A.add_argument('-d', '--dset', default='edflow.debug.ConfigDebugDataset')
    args = A.parse_args()
    dset = args.dset

    return dset


def load_dset(dset, config={}):
    module, cls = dset.rsplit('.', 1)
    dset_class = getattr(importlib.import_module(module, package=None), cls)

    dset = dset_class(config)

    return dset


def isimage(obj):
    return isinstance(obj, np.ndarray) and len(obj.shape) == 3


def istext(obj):
    return isinstance(obj, (int, float, str))


def display_default(obj):
    if isimage(obj):
        return 'Image'
    elif istext(obj):
        return 'Text'
    else:
        return 'None'


def display(key, obj):
    st.subheader(key)
    sel = selector(key, obj)
    if sel == 'Text':
        st.text(obj)

    elif sel == 'Image':
        st.image(obj)

def selector(key, obj):

    options = ['Auto', 'Text', 'Image', 'None']
    idx = options.index(display_default(obj))

    select = st.selectbox('Display {} as'.format(key), options, index=idx)

    return select


def show_example(dset, idx):
    ex = dset[idx]

    walk(ex, display, pass_key=True)

    st.write(ex)


def app():

    dset = process_args()
    dset = load_dset(dset)

    dset = TestDset({})

    st.title('Dataset Explorer')

    side1 = st.sidebar
    side2 = st.sidebar

    idx = side1.slider('Index', 0, len(dset), 0)

    show_example(dset, idx)

    if side2.checkbox('Balloons?!?'):
        st.balloons()


if __name__ == '__main__':
    app()
