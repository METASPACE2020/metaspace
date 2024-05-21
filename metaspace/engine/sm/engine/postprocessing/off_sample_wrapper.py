import json
import logging
from uuid import uuid4

from PIL import Image
from requests import post, get
import numpy as np

from off_sample_orchestrator.orchestrator import Orchestrator, Job

from sm.engine.errors import SMError
from sm.engine import image_storage
from sm.engine.utils.retry_on_exception import retry_on_exception


logger = logging.getLogger('update-daemon')


def make_chunk_gen(items, chunk_size):
    chunk_n = (len(items) - 1) // chunk_size + 1
    chunks = [items[i * chunk_size : (i + 1) * chunk_size] for i in range(chunk_n)]
    for image_path_chunk in chunks:
        yield image_path_chunk


def base64_images_to_doc(images):
    images_doc = {'images': [{'content': content} for content in images]}
    return images_doc


SEL_ION_IMAGES = (
    'select m.id as ann_id, iso_image_ids[1] as img_id '
    'from dataset d '
    'join job j on j.ds_id = d.id '
    'join annotation m on m.job_id = j.id '
    'where d.id = %s and (%s or m.off_sample is null) and iso_image_ids[1] is not NULL '
    'order by m.id '
)
UPD_OFF_SAMPLE = (
    'update annotation as row set off_sample = row2.off_sample::json '
    'from (values %s) as row2(id, off_sample) '
    'where row.id = row2.id; '
)


def numpy_to_pil(a):
    assert a.ndim > 1

    if a.ndim == 2:
        a_min, a_max = a.min(), a.max()
    else:
        a = a[:, :, :3]
        a_min, a_max = a.min(axis=(0, 1)), a.max(axis=(0, 1))

    a = ((a - a_min) / (a_max - a_min) * 255).astype(np.uint8)
    return Image.fromarray(a)


@retry_on_exception(SMError, num_retries=6, retry_wait_params=(10, 10, 5))
def call_api(url='', batch_id=None, doc=None):
    if doc:
        resp = post(url=url, json=doc, timeout=(120, 120))
    else:
        resp = get(url=url)
    if resp.status_code == 200:
        return resp.json()
    logger.info(f'call_api exception, batch_id: {batch_id}')
    raise SMError(resp.content or resp)


def extract_predictions(data):
    flattened_data = []
    for block in data['output']['lithops_results']:
        predictions = block["body"]["predictions"]
        for _, value in predictions.items():
            flattened_data.append({"label": value["label"], "prob": value["prob"]})
    return flattened_data


def make_classify_images(ds_id, services_config):
    def classify(items):
        batch_id = str(uuid4())
        logger.info(f'Classifying chunk of {len(items)} images. ds_id is {ds_id}.')

        images_doc = {
            'inputs': [
                image_storage.get_image_url(image_storage.ISO, ds_id, img_id) for img_id in items
            ]
        }
        images_doc['job_name'] = f'{ds_id} {batch_id}'
        lambda_fexec_args = {
            'runtime': services_config['off_sample_config']['runtime'],
            'runtime_memory': 3008,
            'config': services_config['off_sample_config'],
        }
        orchestrator = Orchestrator(
            fexec_args=lambda_fexec_args, ec2_host_machine=False, initialize=False
        )
        job = Job(
            images_doc['inputs'],
            images_doc['job_name'],
            orchestrator_backend="aws_lambda",
            dynamic_split=False,
        )
        result = orchestrator.run_job(job)
        logger.info(f'Off-sample classification of {len(items)} images')

        return extract_predictions(result)

    return classify


def classify_dataset_ion_images(db, ds, services_config, overwrite_existing=False):
    """Classifies all dataset ion images.

    Args:
        db (sm.engine.db.DB): database connection
        ds (sm.engine.dataset.Dataset): target dataset
        services_config (dict): configuration for services
        overwrite_existing (bool): whether to overwrite existing image classes
    """
    annotations = db.select_with_fields(SEL_ION_IMAGES, (ds.id, overwrite_existing))
    image_ids = [a['img_id'] for a in annotations]

    classify = make_classify_images(ds.id, services_config)
    image_predictions = classify(image_ids)

    rows = [(ann['ann_id'], json.dumps(pred)) for ann, pred in zip(annotations, image_predictions)]
    db.alter_many(UPD_OFF_SAMPLE, rows)
