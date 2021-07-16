import json
from datetime import datetime
import time
from unittest.mock import MagicMock, patch

import pandas as pd

from sm.engine.db import DB
from sm.engine.es_export import (
    ESExporter,
    ESExporterIsobars,
    ESIndexManager,
)
from sm.engine.annotation.isocalc_wrapper import IsocalcWrapper
from sm.engine.molecular_db import MolecularDB
from .utils import create_test_molecular_db


def wait_for_es(sec: float = 1):
    time.sleep(sec)


def test_index_ds_works(sm_config, test_db, es_dsl_search, sm_index, ds_config, metadata):
    ds_id = '2000-01-01_00h00m'
    upload_dt = datetime.now().isoformat()
    last_finished = '2017-01-01 00:00:00'
    iso_image_ids = ['iso_img_id_1', 'iso_img_id_2']
    annotation_stats = json.dumps(
        {
            'chaos': 1,
            'spatial': 1,
            'spectral': 1,
            'total_iso_ints': 100,
            'min_iso_ints': 0,
            'max_iso_ints': 100,
        }
    )

    db = DB()
    db.insert(
        "INSERT INTO dataset(id, name, input_path, config, metadata, upload_dt, status, "
        "status_update_dt, is_public, acq_geometry, ion_thumbnail) "
        "VALUES (%s, 'ds_name', 'ds_input_path', %s, %s, %s, 'ds_status', %s, true, '{}', %s)",
        [[ds_id, json.dumps(ds_config), json.dumps(metadata), upload_dt, upload_dt, 'thumb-id']],
    )
    moldb = create_test_molecular_db()
    (job_id,) = db.insert_return(
        "INSERT INTO job(ds_id, moldb_id, status, start, finish) "
        "VALUES (%s, %s, 'job_status', %s, %s) RETURNING id",
        rows=[(ds_id, moldb.id, last_finished, last_finished)],
    )
    (user_id,) = db.insert_return(
        "INSERT INTO graphql.user (email, name, role) "
        "VALUES ('email', 'user_name', 'user') RETURNING id",
        [[]],
    )
    (group_id,) = db.insert_return(
        "INSERT INTO graphql.group (name, short_name) VALUES ('group name', 'grp') RETURNING id",
        [[]],
    )
    db.insert(
        "INSERT INTO graphql.dataset(id, user_id, group_id) VALUES (%s, %s, %s)",
        [[ds_id, user_id, group_id]],
    )
    ion_id1, ion_id2 = db.insert_return(
        "INSERT INTO graphql.ion(ion, formula, chem_mod, neutral_loss, adduct, charge, ion_formula) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
        [
            ['H2O-H+O-H+H', 'H2O', '-H+O', '-H', '+H', 1, 'HO2'],
            ['Au+H', 'Au', '', '', '+H', 1, 'HAu'],
        ],
    )
    db.insert(
        "INSERT INTO annotation(job_id, formula, chem_mod, neutral_loss, adduct, "
        "msm, fdr, stats, iso_image_ids, ion_id) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        [
            [job_id, 'H2O', '-H+O', '-H', '+H', 1, 0.1, annotation_stats, iso_image_ids, ion_id1],
            [job_id, 'Au', '', '', '+H', 1, 0.05, annotation_stats, iso_image_ids, ion_id2],
        ],
    )

    isocalc_mock = MagicMock(IsocalcWrapper)
    isocalc_mock.centroids = lambda formula: {
        'H2O+H': ([100.0, 200.0], None),
        'H2O-H+O-H+H': ([100.0, 200.0, 300.0], None),
        'Au+H': ([10.0, 20.0], None),
    }[formula]
    isocalc_mock.mass_accuracy_bounds = lambda mzs: (mzs, mzs)

    with patch(
        'sm.engine.es_export.molecular_db.fetch_molecules',
        return_value=pd.DataFrame(
            [('H2O', 'mol_id', 'mol_name'), ('Au', 'mol_id', 'mol_name')],
            columns=['formula', 'mol_id', 'mol_name'],
        ),
    ):
        es_exp = ESExporter(db, sm_config)
        es_exp.delete_ds(ds_id)
        es_exp.index_ds(
            ds_id=ds_id,
            moldb=moldb,
            isocalc=isocalc_mock,
        )

    wait_for_es(sec=1.5)

    ds_d = (
        es_dsl_search.filter('term', _type='dataset')
        .execute()
        .to_dict()['hits']['hits'][0]['_source']
    )
    expected_ds_fields = {
        'ds_last_finished': last_finished,
        'ds_config': ds_config,
        'ds_adducts': ds_config['isotope_generation']['adducts'],
        'ds_moldb_ids': ds_config['database_ids'],
        'ds_chem_mods': [],
        'ds_neutral_losses': [],
        'ds_project_ids': [],
        'ds_project_names': [],
        'ds_meta': metadata,
        'ds_status': 'ds_status',
        'ds_status_update_dt': upload_dt,
        'ds_name': 'ds_name',
        'ds_input_path': 'ds_input_path',
        'ds_id': ds_id,
        'ds_upload_dt': upload_dt,
        'ds_is_public': True,
        'ds_submitter_email': 'email',
        'ds_submitter_id': user_id,
        'ds_submitter_name': 'user_name',
        'ds_group_approved': False,
        'ds_group_id': group_id,
        'ds_group_name': 'group name',
        'ds_group_short_name': 'grp',
    }
    assert ds_d == {
        **expected_ds_fields,
        'ds_acq_geometry': {},
        'annotation_counts': [
            {
                'db': {'id': moldb.id, 'name': moldb.name},
                'counts': [
                    {'level': 5, 'n': 1},
                    {'level': 10, 'n': 2},
                    {'level': 20, 'n': 2},
                    {'level': 50, 'n': 2},
                ],
            }
        ],
    }
    ann_1_d = (
        es_dsl_search.filter('term', formula='H2O')
        .execute()
        .to_dict()['hits']['hits'][0]['_source']
    )
    assert ann_1_d == {
        **expected_ds_fields,
        'pattern_match': 1.0,
        'image_corr': 1.0,
        'fdr': 0.1,
        'chaos': 1.0,
        'formula': 'H2O',
        'min_iso_ints': 0,
        'msm': 1.0,
        'ion': 'H2O-H+O-H+H+',
        'ion_formula': 'HO2',
        'total_iso_ints': 100,
        'centroid_mzs': [100.0, 200.0, 300.0],
        'iso_image_ids': ['iso_img_id_1', 'iso_img_id_2'],
        'iso_image_urls': [
            f'http://localhost:9000/{sm_config["image_storage"]["bucket"]}/iso/{ds_id}/iso_img_id_1',
            f'http://localhost:9000/{sm_config["image_storage"]["bucket"]}/iso/{ds_id}/iso_img_id_2',
        ],
        'isobars': [],
        'isomer_ions': [],
        'polarity': '+',
        'job_id': 1,
        'max_iso_ints': 100,
        'adduct': '+H',
        'neutral_loss': '-H',
        'chem_mod': '-H+O',
        'annotation_counts': [],
        'comp_names': ['mol_name'],
        'comps_count_with_isomers': 1,
        'db_id': moldb.id,
        'db_name': moldb.name,
        'db_version': moldb.version,
        'mz': 100.0,
        'comp_ids': ['mol_id'],
        'annotation_id': 1,
        'off_sample_label': None,
        'off_sample_prob': None,
    }
    ann_2_d = (
        es_dsl_search.filter('term', formula='Au').execute().to_dict()['hits']['hits'][0]['_source']
    )
    assert ann_2_d == {
        **expected_ds_fields,
        'pattern_match': 1.0,
        'image_corr': 1.0,
        'fdr': 0.05,
        'chaos': 1.0,
        'formula': 'Au',
        'min_iso_ints': 0,
        'msm': 1.0,
        'ion': 'Au+H+',
        'ion_formula': 'HAu',
        'total_iso_ints': 100,
        'centroid_mzs': [10.0, 20.0],
        'iso_image_ids': ['iso_img_id_1', 'iso_img_id_2'],
        'iso_image_urls': [
            f'http://localhost:9000/{sm_config["image_storage"]["bucket"]}/iso/{ds_id}/iso_img_id_1',
            f'http://localhost:9000/{sm_config["image_storage"]["bucket"]}/iso/{ds_id}/iso_img_id_2',
        ],
        'isobars': [],
        'isomer_ions': [],
        'polarity': '+',
        'job_id': 1,
        'max_iso_ints': 100,
        'adduct': '+H',
        'neutral_loss': '',
        'chem_mod': '',
        'annotation_counts': [],
        'comp_names': ['mol_name'],
        'comps_count_with_isomers': 1,
        'db_id': moldb.id,
        'db_name': moldb.name,
        'db_version': moldb.version,
        'mz': 10.0,
        'comp_ids': ['mol_id'],
        'annotation_id': 2,
        'off_sample_label': None,
        'off_sample_prob': None,
    }


def test_add_isomer_fields_to_anns():
    ann_docs = [
        {'ion': 'H2O+H-H-', 'ion_formula': 'H2O', 'comp_ids': ['1']},
        {'ion': 'H3O-H-', 'ion_formula': 'H2O', 'comp_ids': ['2', '3']},
        {'ion': 'H3O+CO2-CO2-H-', 'ion_formula': 'H2O', 'comp_ids': ['2', '3', '4']},
        {'ion': 'H2O-H-', 'ion_formula': 'H1O', 'comp_ids': ['4']},
    ]

    ESExporter._add_isomer_fields_to_anns(ann_docs)

    isomer_ions_fields = [doc['isomer_ions'] for doc in ann_docs]
    comps_count_fields = [doc['comps_count_with_isomers'] for doc in ann_docs]
    assert isomer_ions_fields == [
        ['H3O-H-', 'H3O+CO2-CO2-H-'],
        ['H2O+H-H-', 'H3O+CO2-CO2-H-'],
        ['H2O+H-H-', 'H3O-H-'],
        [],
    ]

    assert comps_count_fields == [4, 4, 4, 1]


def test_add_isobar_fields_to_anns(ds_config):
    ann_docs = [
        {
            'annotation_id': 'Base annotation',
            'centroid_mzs': [100, 101, 102, 103],
            'iso_image_urls': ['img1', 'img2', 'img3', 'img4'],
            'msm': 0.5,
            'ion': 'H1+',
            'ion_formula': 'H1',
        },
        {
            'annotation_id': "Base's 1st centroid overlaps 1st",
            'centroid_mzs': [100.0002, 101.1, 102.1, 103.1],
            'iso_image_urls': ['img1', 'img2', 'img3', 'img4'],
            'msm': 0.6,
            'ion': 'H2+',
            'ion_formula': 'H2',
        },
        {
            'annotation_id': "Base's 1st centroid overlaps 2nd (shouldn't be reported)",
            'centroid_mzs': [98, 100.0002, 101.2, 102.2],
            'iso_image_urls': ['img1', 'img2', 'img3', 'img4'],
            'msm': 0.7,
            'ion': 'H3+',
            'ion_formula': 'H3',
        },
        {
            'annotation_id': "Base's 2nd and 3rd centroid overlap 3rd and 4th",
            'centroid_mzs': [96, 97, 101, 102],
            'iso_image_urls': ['img1', 'img2', 'img3', 'img4'],
            'msm': 0.8,
            'ion': 'H4+',
            'ion_formula': 'H4',
        },
    ]
    isocalc = IsocalcWrapper(ds_config)

    ESExporterIsobars.add_isobar_fields_to_anns(ann_docs, isocalc)

    isobar_fields = dict((i, doc['isobars']) for i, doc in enumerate(ann_docs))
    assert isobar_fields == {
        0: [
            {'ion': 'H2+', 'ion_formula': 'H2', 'msm': 0.6, 'peak_ns': [(1, 1)]},
            {'ion': 'H4+', 'ion_formula': 'H4', 'msm': 0.8, 'peak_ns': [(2, 3), (3, 4)]},
        ],
        1: [{'ion': 'H1+', 'ion_formula': 'H1', 'msm': 0.5, 'peak_ns': [(1, 1)]}],
        2: [],
        3: [{'ion': 'H1+', 'ion_formula': 'H1', 'msm': 0.5, 'peak_ns': [(3, 2), (4, 3)]}],
    }


def test_delete_ds__one_db_ann_only(sm_config, test_db, es, sm_index):
    moldb = MolecularDB(0, 'HMDB', '2016')
    moldb2 = MolecularDB(1, 'ChEBI', '2016')

    index = sm_config['elasticsearch']['index']
    es.create(
        index=index,
        doc_type='annotation',
        id='id1',
        body={
            'ds_id': 'dataset1',
            'db_id': moldb.id,
            'db_name': moldb.name,
            'db_version': moldb.version,
        },
    )
    es.create(
        index=index,
        doc_type='annotation',
        id='id2',
        body={
            'ds_id': 'dataset1',
            'db_id': moldb2.id,
            'db_name': moldb2.name,
            'db_version': moldb2.version,
        },
    )
    es.create(
        index=index,
        doc_type='annotation',
        id='id3',
        body={
            'ds_id': 'dataset2',
            'db_id': moldb.id,
            'db_name': moldb.name,
            'db_version': moldb.version,
        },
    )
    es.create(
        index=index,
        doc_type='dataset',
        id='id4',
        body={
            'ds_id': 'dataset1',
            'db_id': moldb.id,
            'db_name': moldb.name,
            'db_version': moldb.version,
        },
    )

    wait_for_es(sec=1)

    db_mock = MagicMock(spec=DB)
    es_exporter = ESExporter(db_mock, sm_config)
    es_exporter.delete_ds(ds_id='dataset1', moldb=moldb)

    wait_for_es(sec=1)

    body = {'query': {'bool': {'filter': []}}}
    body['query']['bool']['filter'] = [
        {'term': {'ds_id': 'dataset1'}},
        {'term': {'db_id': moldb.id}},
    ]
    assert es.count(index=index, doc_type='annotation', body=body)['count'] == 0
    body['query']['bool']['filter'] = [
        {'term': {'ds_id': 'dataset1'}},
        {'term': {'db_id': moldb2.id}},
    ]
    assert es.count(index=index, doc_type='annotation', body=body)['count'] == 1
    body['query']['bool']['filter'] = [
        {'term': {'ds_id': 'dataset2'}},
        {'term': {'db_id': moldb.id}},
    ]
    assert es.count(index=index, doc_type='annotation', body=body)['count'] == 1
    body['query']['bool']['filter'] = [
        {'term': {'ds_id': 'dataset1'}},
        {'term': {'_type': 'dataset'}},
    ]
    assert es.count(index=index, doc_type='dataset', body=body)['count'] == 1


def test_delete_ds__completely(sm_config, test_db, es, sm_index):
    moldb = MolecularDB(0, 'HMDB', '2016')
    moldb2 = MolecularDB(1, 'ChEBI', '2016')

    index = sm_config['elasticsearch']['index']
    es.create(
        index=index,
        doc_type='annotation',
        id='id1',
        body={
            'ds_id': 'dataset1',
            'db_id': moldb.id,
            'db_name': moldb.name,
            'db_version': moldb.version,
        },
    )
    es.create(
        index=index,
        doc_type='annotation',
        id='id2',
        body={
            'ds_id': 'dataset1',
            'db_id': moldb2.id,
            'db_name': moldb2.name,
            'db_version': moldb2.version,
        },
    )
    es.create(
        index=index,
        doc_type='annotation',
        id='id3',
        body={
            'ds_id': 'dataset2',
            'db_id': moldb.id,
            'db_name': moldb.name,
            'db_version': moldb.version,
        },
    )
    es.create(
        index=index,
        doc_type='dataset',
        id='dataset1',
        body={
            'ds_id': 'dataset1',
            'db_id': moldb.id,
            'db_name': moldb.name,
            'db_version': moldb.version,
        },
    )

    wait_for_es(sec=1)

    db_mock = MagicMock(spec=DB)

    es_exporter = ESExporter(db_mock, sm_config)
    es_exporter.delete_ds(ds_id='dataset1')

    wait_for_es(sec=1)

    body = {'query': {'bool': {'filter': []}}}
    body['query']['bool']['filter'] = [
        {'term': {'ds_id': 'dataset1'}},
        {'term': {'db_id': moldb.id}},
    ]
    assert es.count(index=index, doc_type='annotation', body=body)['count'] == 0
    body['query']['bool']['filter'] = [
        {'term': {'ds_id': 'dataset1'}},
        {'term': {'db_id': moldb2.id}},
    ]
    assert es.count(index=index, doc_type='annotation', body=body)['count'] == 0
    body['query']['bool']['filter'] = [
        {'term': {'ds_id': 'dataset2'}},
        {'term': {'db_id': moldb.id}},
    ]
    assert es.count(index=index, doc_type='annotation', body=body)['count'] == 1
    body['query']['bool']['filter'] = [
        {'term': {'ds_id': 'dataset1'}},
        {'term': {'_type': 'dataset'}},
    ]
    assert es.count(index=index, doc_type='dataset', body=body)['count'] == 0


def test_update_ds_works_for_all_fields(sm_config, test_db, es, sm_index, es_dsl_search):
    update = {
        'name': 'new_ds_name',
        'submitter_id': 'new_ds_submitter_id',
        'group_id': 'new_ds_group_id',
        'projects_ids': ['proj_id1', 'proj_id2'],
        'is_public': True,
    }

    index = sm_config['elasticsearch']['index']
    es.create(
        index=index,
        doc_type='annotation',
        id='id1',
        body={
            'ds_id': 'dataset1',
            'ds_name': 'ds_name',
            'ds_submitter_id': 'ds_submitter',
            'ds_group_id': 'ds_group_id',
            'ds_project_ids': [],
            'ds_is_public': False,
        },
    )
    es.create(
        index=index,
        doc_type='dataset',
        id='dataset1',
        body={
            'ds_id': 'dataset1',
            'ds_name': 'ds_name',
            'ds_submitter_id': 'ds_submitter_id',
            'ds_group_id': 'ds_group_id',
            'ds_projects_ids': [],
            'ds_is_public': False,
        },
    )
    wait_for_es(sec=1)

    db_mock = MagicMock(spec=DB)
    db_mock.select_with_fields.return_value = [
        {
            'ds_name': 'new_ds_name',
            'ds_submitter_id': 'new_ds_submitter_id',
            'ds_submitter_name': 'submitter_name',
            'ds_submitter_email': 'submitter_email',
            'ds_group_id': 'new_ds_group_id',
            'ds_group_name': 'group_name',
            'ds_group_approved': True,
            'ds_group_short_name': 'group_short_name',
            'ds_projects_ids': ['proj_id1', 'proj_id2'],
            'ds_is_public': True,
        }
    ]

    es_exporter = ESExporter(db_mock, sm_config)
    es_exporter.update_ds('dataset1', fields=list(update.keys()))
    wait_for_es(sec=1)

    ds_doc = (
        es_dsl_search.filter('term', _type='dataset')
        .execute()
        .to_dict()['hits']['hits'][0]['_source']
    )
    for k, v in update.items():
        assert v == ds_doc[f'ds_{k}']

    ann_doc = (
        es_dsl_search.filter('term', _type='annotation')
        .execute()
        .to_dict()['hits']['hits'][0]['_source']
    )
    for k, v in update.items():
        assert v == ann_doc[f'ds_{k}']


def test_rename_index_works(sm_config, test_db):
    es_config = sm_config['elasticsearch']
    alias = es_config['index']
    es_man = ESIndexManager(es_config)

    es_man.create_index('{}-yin'.format(alias))
    es_man.remap_alias('{}-yin'.format(alias), alias=alias)

    assert es_man.exists_index(alias)
    assert es_man.exists_index('{}-yin'.format(alias))
    assert not es_man.exists_index('{}-yang'.format(alias))

    es_man.create_index('{}-yang'.format(alias))
    es_man.remap_alias('{}-yang'.format(alias), alias=alias)

    assert es_man.exists_index(alias)
    assert es_man.exists_index('{}-yang'.format(alias))
    assert es_man.exists_index('{}-yin'.format(alias))


def test_internal_index_name_return_valid_values(sm_config):
    es_config = sm_config['elasticsearch']
    alias = es_config['index']
    es_man = ESIndexManager(es_config)

    assert es_man.internal_index_name(alias) in ['{}-yin'.format(alias), '{}-yang'.format(alias)]
