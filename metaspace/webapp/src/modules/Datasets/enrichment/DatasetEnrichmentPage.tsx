import { computed, defineComponent, reactive } from 'vue'
import { useQuery } from '@vue/apollo-composable'
import { getDatasetByIdQuery, GetDatasetByIdQuery, getDatasetEnrichmentQuery } from '../../../api/dataset'
import { DatasetEnrichmentChart } from './DatasetEnrichmentChart'
import { DatasetEnrichmentTable } from './DatasetEnrichmentTable'
import { getEnrichedMolDatabasesQuery } from '../../../api/enrichmentdb'
import FilterPanel from '../../Filters/FilterPanel.vue'
import { uniqBy } from 'lodash-es'
import './DatasetEnrichmentPage.scss'
import { useRoute, useRouter } from 'vue-router'
import { useStore } from 'vuex'
import { ElIcon } from '../../../lib/element-plus'
import { Loading } from '@element-plus/icons-vue'
import gql from 'graphql-tag'
import safeJsonParse from '../../../lib/safeJsonParse'

interface DatasetEnrichmentPageState {
  offset: number
  pageSize: number
  sortedData: any
}

export default defineComponent({
  name: 'DatasetEnrichmentPage',
  props: {
    className: {
      type: String,
      default: 'dataset-enrichment',
    },
  },
  setup() {
    const router = useRouter()
    const route = useRoute()
    const store = useStore()
    const state = reactive<DatasetEnrichmentPageState>({
      offset: 0,
      pageSize: 15,
      sortedData: undefined,
    })
    const fetchImageViewerSnapshot = gql`
      query fetchImageViewerSnapshot($id: String!, $datasetId: String!) {
        imageViewerSnapshot(id: $id, datasetId: $datasetId) {
          snapshot
        }
      }
    `
    const snapQueryOptions = reactive({ enabled: false, fetchPolicy: 'no-cache' as const })
    const enrichmentQueryOptions = reactive({ enabled: false, fetchPolicy: 'cache-first' as const })

    const datasetId = computed(() => route.params.dataset_id)

    const snapshotId = computed(() => route.query.viewId)
    const { onResult: handleSettingsLoad } = useQuery(
      fetchImageViewerSnapshot,
      {
        id: snapshotId,
        datasetId: datasetId,
      },
      snapQueryOptions
    )

    handleSettingsLoad(async (result) => {
      const snapFilter = safeJsonParse(result?.data?.imageViewerSnapshot?.snapshot)

      if (snapFilter) {
        const filter = Object.assign(store.getters.filter, snapFilter)

        await store.commit('updateFilter', filter)
      }

      enrichmentQueryOptions.enabled = true
    })

    const { result: datasetResult, onResult: handleDatasetLoad } = useQuery<GetDatasetByIdQuery>(getDatasetByIdQuery, {
      id: datasetId.value,
    })

    handleDatasetLoad(async (result) => {
      const filter = Object.assign({}, store.getters.filter)

      if (!filter.ontology) {
        const ontologyDatabases: any = result?.data?.dataset?.ontologyDatabases || []
        if (ontologyDatabases.length > 0) {
          filter.ontology = ontologyDatabases[0].id
          store.commit('updateFilter', filter)
        }
      }

      if (route.query.viewId) {
        snapQueryOptions.enabled = true
      } else {
        enrichmentQueryOptions.enabled = true
      }
    })

    const dataset = computed(() => (datasetResult.value != null ? datasetResult.value.dataset : null))
    const { result: databasesResult, onResult: handleMolDbLoad } = useQuery<any>(getEnrichedMolDatabasesQuery, {
      id: datasetId.value,
    })

    handleMolDbLoad(async (result: any) => {
      const filter = Object.assign({}, store.getters.filter)

      // set default db filter if not selected
      if (!filter.database) {
        filter.database = result?.data?.allEnrichedMolDatabases[0]?.id
        store.commit('updateFilter', filter)
      }
    })

    const databases = computed(() =>
      databasesResult.value != null ? databasesResult.value.allEnrichedMolDatabases : null
    )
    const { result: enrichmentResult, loading: enrichmentLoading } = useQuery(
      getDatasetEnrichmentQuery,
      computed(() => ({
        id: datasetId.value,
        ontologyId: store.getters.gqlAnnotationFilter.ontologyId,
        colocalizedWith: store.getters.gqlAnnotationFilter.colocalizedWith,
        dbId: store.getters.gqlAnnotationFilter.databaseId,
        fdr: store.getters.gqlAnnotationFilter.fdrLevel,
        pValue:
          store.getters.gqlAnnotationFilter.pValue === null || store.getters.gqlAnnotationFilter.pValue === undefined
            ? undefined
            : store.getters.gqlAnnotationFilter.pValue,
        offSample:
          store.getters.gqlAnnotationFilter.offSample === null ||
          store.getters.gqlAnnotationFilter.offSample === undefined
            ? undefined
            : !!store.getters.gqlAnnotationFilter.offSample,
      })),
      enrichmentQueryOptions
    )

    const enrichment = computed(() => {
      if (enrichmentResult.value) {
        return enrichmentResult.value.lipidEnrichment
      }
      return null
    })

    const handlePageChange = (offset: number) => {
      state.offset = offset
    }

    const handleSizeChange = (pageSize: number) => {
      state.pageSize = pageSize
    }

    const handleSortChange = (newData: number) => {
      state.sortedData = newData
    }

    const handleItemClick = (item: any) => {
      const dbId = store.getters.gqlAnnotationFilter.databaseId
      const fdr = store.getters.gqlAnnotationFilter.fdrLevel

      const routeData = router.resolve({
        name: 'annotations',
        query: {
          term: item?.termId,
          ds: datasetId.value,
          db_id: dbId,
          mol_class: store.getters.gqlAnnotationFilter.ontologyId,
          fdr,
          feat: 'enrichment',
        },
      })
      window.open(routeData.href, '_blank')
    }

    return () => {
      const dataStart = (state.offset - 1) * state.pageSize
      const dataEnd = (state.offset - 1) * state.pageSize + state.pageSize
      const data = enrichment.value || []
      const usedData = state.sortedData ? state.sortedData : data
      const pagedData = usedData.slice(dataStart, dataEnd)
      const databaseOptions: any = databases.value || []
      const filename: string = `${dataset.value?.name}_${(databases.value || []).find(
        (database: any) => database.id === store.getters.gqlAnnotationFilter.databaseId
      )?.name}`.replace(/\./g, '_')
      const ontologyDatabases: any = dataset.value?.ontologyDatabases || []

      return (
        <div class="dataset-enrichment-page">
          {databases.value && (
            <FilterPanel
              class="w-full"
              level="enrichment"
              fixedOptions={{
                database: uniqBy(databaseOptions, 'id'),
                ontology: uniqBy(ontologyDatabases, 'id'),
              }}
            />
          )}
          {enrichmentLoading.value && (
            <div class="w-full h-full flex items-center justify-center">
              <ElIcon class="is-loading">
                <Loading />
              </ElIcon>
              <span>Loading...</span>
            </div>
          )}
          {!enrichmentLoading.value && (
            <div class={'dataset-enrichment-wrapper md:w-1/2 w-full'}>
              <DatasetEnrichmentTable
                dsName={dataset.value?.name}
                data={data}
                filename={`${dataset.value?.name}_${databases?.value?.find(
                  (database: any) => database.id === store.getters.gqlAnnotationFilter.databaseId
                )?.name}_enrichment.csv`}
                onPageChange={handlePageChange}
                onSizeChange={handleSizeChange}
                onSortChange={handleSortChange}
              />
            </div>
          )}
          {!enrichmentLoading.value && (
            <div class={'dataset-enrichment-wrapper text-center md:w-1/2 w-full'}>
              {dataset.value?.name} - terms enrichment
              {!(!data || (data || []).length === 0) && (
                <DatasetEnrichmentChart
                  data={pagedData}
                  onItemSelected={handleItemClick}
                  filename={`Enrichment_${filename}_LION`}
                />
              )}
            </div>
          )}
        </div>
      )
    }
  },
})
