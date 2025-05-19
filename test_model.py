import dtlpy as dl
from model_adapter import Adapter
import json

if __name__ == "__main__":

    dl.setenv('prod')
    item = dl.items.get(item_id='6783f473c4d6ff8a61ed8f16')
    project = item.project
    with open("dataloop.json", "r") as f:
        dpk = dl.Dpk.from_json(
            _json=json.load(f),
            client_api=dl.client_api,
        )
    model_entity = dl.Model.from_json(
        _json=dpk.components.models[0],
        client_api=dl.client_api,
        project=project,
        package=dpk,
    )
    adapter = Adapter(
        model_entity=model_entity,
    )
    batch = adapter.prepare_item_func(item)
    batch_annotations = adapter.predict([batch])
    item.annotations.upload(batch_annotations[0])
