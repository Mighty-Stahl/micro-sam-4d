import numpy as np

from micro_sam.sam_annotator.annotator_4d import MicroSAM4DAnnotator


class FakeLayer:
    def __init__(self, data, name):
        self.data = np.array(data)
        self.name = name
        self.scale = None

    def refresh(self):
        pass

    def new_colormap(self):
        pass


class DummyEvent:
    def connect(self, *args, **kwargs):
        pass


class FakeEvents:
    def __init__(self):
        self.inserted = DummyEvent()
        self.removed = DummyEvent()


class LayerStore(dict):
    def __init__(self):
        super().__init__()
        self.events = FakeEvents()

    def add_image(self, data, name=None, **kwargs):
        name = name or "image"
        lay = FakeLayer(data, name)
        self[name] = lay
        return lay

    def add_labels(self, data, name=None, **kwargs):
        name = name or "labels"
        lay = FakeLayer(data, name)
        self[name] = lay
        return lay

    def add_points(self, data, name=None, **kwargs):
        name = name or "points"
        lay = FakeLayer(data, name)
        self[name] = lay
        return lay

    def remove(self, layer):
        if isinstance(layer, FakeLayer) and layer.name in self:
            del self[layer.name]


class FakeDims:
    def __init__(self):
        self.ndim = None


class FakeViewer:
    def __init__(self):
        self.layers = LayerStore()
        self.dims = FakeDims()

    def add_image(self, data, name=None, **kwargs):
        return self.layers.add_image(data, name=name, **kwargs)

    def add_labels(self, data, name=None, **kwargs):
        return self.layers.add_labels(data, name=name, **kwargs)

    def add_points(self, data, name=None, **kwargs):
        return self.layers.add_points(data, name=name, **kwargs)


def test_4d_smoke():
    T, Z, Y, X = 3, 4, 16, 16
    img = np.zeros((T, Z, Y, X), dtype=np.uint8)
    viewer = FakeViewer()

    # construct annotator without running its __init__ (which creates UI widgets)
    annot = object.__new__(MicroSAM4DAnnotator)
    annot._viewer = viewer

    # initialize 4D image state
    annot.update_image(img)
    assert annot.n_timesteps == T

    # create dummy segmentation for current timestep
    seg = np.ones((Z, Y, X), dtype=np.uint32)
    # set the per-timestep current_object layer data
    viewer.layers["current_object"].data = seg.copy()
    annot.save_current_object_to_4d()
    assert np.array_equal(annot.current_object_4d[0], seg)

    # commit segmentation
    annot.commit_segmentation(seg)
    assert np.array_equal(annot.segmentation_4d[0], seg)

    # go to next timestep and check current_timestep and that layer data corresponds to slice
    annot.next_timestep()
    assert annot.current_timestep == 1

    # write a seg to timestep 1 and commit
    seg2 = np.full((Z, Y, X), 2, dtype=np.uint32)
    viewer.layers["current_object"].data = seg2.copy()
    annot.save_current_object_to_4d()
    annot.commit_segmentation(seg2)
    assert np.array_equal(annot.segmentation_4d[1], seg2)
