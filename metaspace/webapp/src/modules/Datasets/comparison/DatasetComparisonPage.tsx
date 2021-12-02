import { Collapse, CollapseItem } from '../../../lib/element-ui'
import {
  computed,
  defineComponent,
  onMounted,
  reactive,
  ref, watch,
  watchEffect,
} from '@vue/composition-api'
import { useQuery } from '@vue/apollo-composable'
import { annotationListQuery, comparisonAnnotationListQuery } from '../../../api/annotation'
import safeJsonParse from '../../../lib/safeJsonParse'
import RelatedMolecules from '../../Annotations/annotation-widgets/RelatedMolecules.vue'
import ImageSaver from '../../ImageViewer/ImageSaver.vue'
import { DatasetComparisonAnnotationTable } from './DatasetComparisonAnnotationTable'
import { DatasetComparisonGrid } from './DatasetComparisonGrid'
import gql from 'graphql-tag'
import FilterPanel from '../../Filters/FilterPanel.vue'
import config from '../../../lib/config'
import {
  DatasetListItemWithDiagnostics,
  datasetListItemsWithDiagnosticsQuery,
} from '../../../api/dataset'
import MainImageHeader from '../../Annotations/annotation-widgets/default/MainImageHeader.vue'
import CandidateMoleculesPopover from '../../Annotations/annotation-widgets/CandidateMoleculesPopover.vue'
import MolecularFormula from '../../../components/MolecularFormula'
import CopyButton from '../../../components/CopyButton.vue'
import { DatasetComparisonShareLink } from './DatasetComparisonShareLink'
import { groupBy, isEqual, uniqBy, uniq } from 'lodash-es'
import { readNpy } from '../../../lib/npyHandler'

interface GlobalImageSettings {
  resetViewPort: boolean
  isNormalized: boolean
  scaleBarColor: string
  scaleType: string
  colormap: string
  selectedLockTemplate: string | null
  globalLockedIntensities: [number | undefined, number | undefined]
  showOpticalImage: boolean
}

interface DatasetComparisonPageProps {
  className: string
  defaultImagePosition: any
}

interface DatasetComparisonPageState {
  selectedAnnotation: number
  gridState: any
  annotations: any
  datasets: any
  globalImageSettings: GlobalImageSettings
  grid: any
  nCols: number
  nRows: number
  annotationData: any
  refsLoaded: boolean
  showViewer: boolean
  annotationLoading: boolean
  isLoading: any
  collapse: string[]
  databaseOptions: any
  normalizationData: any
  offset: number
  rawAnnotations: any
  processedAnnotations: any
}

const CHUNK_SIZE = 1000

export default defineComponent<DatasetComparisonPageProps>({
  name: 'DatasetComparisonPage',
  props: {
    className: {
      type: String,
      default: 'dataset-comparison',
    },
    defaultImagePosition: {
      type: Object,
      default: () => ({
        zoom: 1,
        xOffset: 0,
        yOffset: 0,
      }),
    },
  },

  // @ts-ignore
  setup(props, { refs, root }) {
    const fetchImageViewerSnapshot = gql`query fetchImageViewerSnapshot($id: String!, $datasetId: String!) {
      imageViewerSnapshot(id: $id, datasetId: $datasetId) {
        snapshot
      }
    }`
    const { $route, $store } = root
    const gridNode = ref(null)
    const imageGrid = ref(null)
    const state = reactive<DatasetComparisonPageState>({
      selectedAnnotation: -1,
      gridState: {},
      databaseOptions: undefined,
      globalImageSettings: {
        resetViewPort: false,
        isNormalized: false,
        scaleBarColor: '#000000',
        scaleType: 'linear',
        colormap: 'Viridis',
        showOpticalImage: false,
        selectedLockTemplate: null,
        globalLockedIntensities: [undefined, undefined],
      },
      annotations: undefined,
      datasets: [],
      collapse: ['images'],
      grid: undefined,
      nCols: 0,
      nRows: 0,
      annotationData: {},
      refsLoaded: false,
      showViewer: false,
      annotationLoading: true,
      isLoading: false,
      normalizationData: {},
      offset: 0,
      rawAnnotations: [],
      processedAnnotations: [],
    })
    const { dataset_id: sourceDsId } = $route.params
    const { viewId: snapshotId } = $route.query
    const {
      result: settingsResult,
    } = useQuery<any>(fetchImageViewerSnapshot, {
      id: snapshotId,
      datasetId: sourceDsId,
    })

    const gridSettings = computed(() => settingsResult.value != null
      ? settingsResult.value.imageViewerSnapshot : null)

    const queryVariables = () => {
      const filter = $store.getters.gqlAnnotationFilter
      const dFilter = $store.getters.gqlDatasetFilter
      const colocalizationCoeffFilter = $store.getters.gqlColocalizationFilter
      const query = $store.getters.ftsQuery

      return {
        filter,
        dFilter,
        query,
        colocalizationCoeffFilter,
        countIsomerCompounds: config.features.isomers,
      }
    }

    const annotationQueryOptions = reactive({ enabled: false, fetchPolicy: 'no-cache' as const })
    const dsQueryOptions = reactive({ enabled: false, fetchPolicy: 'no-cache' as const })
    const annotationQueryVars = computed(() => ({
      ...queryVariables(),
      dFilter: { ...queryVariables().dFilter, ids: Object.values(state.grid || {}).join('|') },
      limit: CHUNK_SIZE,
      offset: state.offset,
    }))

    const {
      result: annotationsResult,
      loading: annotationsLoading,
      onResult: onAllAnnotationsResult,
    } = useQuery<any>(annotationListQuery, annotationQueryVars, annotationQueryOptions)

    onAllAnnotationsResult(async(result) => {
      let rawAnnotations = state.rawAnnotations
      if (result && result.data) {
        if (state.offset === 0) {
          rawAnnotations = []
        }
        rawAnnotations = rawAnnotations.concat(result.data.allAnnotations)

        state.rawAnnotations = rawAnnotations
        if (state.offset < result.data.countAnnotations) {
          state.offset += CHUNK_SIZE
        } else {
          annotationQueryOptions.enabled = false
          state.offset = 0
          const annotationsByIon = groupBy(state.rawAnnotations, 'ion')
          const processedAnnotations = Object.keys(annotationsByIon).map((ion: string) => {
            const annotations = annotationsByIon[ion]
            const dbId = annotations[0].databaseDetails.id
            const datasetIds = uniq(annotations.map((annotation: any) => annotation.dataset.id))

            return {
              ion,
              dbId,
              datasetIds,
              annotations,
            }
          })

          state.annotations = processedAnnotations
          dsQueryOptions.enabled = true
          state.annotationLoading = false
        }
      }
    })

    const {
      result: datasetsResult,
      onResult: onDatasetsResult,
    } = useQuery<{allDatasets: DatasetListItemWithDiagnostics[]}>(datasetListItemsWithDiagnosticsQuery,
      () => ({
        dFilter: {
          ...queryVariables().dFilter,
          ids:
            gridSettings.value ? Object.values((safeJsonParse(gridSettings.value.snapshot) || {}).grid || {})
              .join('|') : '',
        },
      }), dsQueryOptions)

    onDatasetsResult(async(result) => {
      const normalizationData : any = {}
      // calculate normalization
      if (result && result.data && result.data.allDatasets) {
        let databases : any[] = []

        for (let i = 0; i < result.data.allDatasets.length; i++) {
          const dataset : any = result.data.allDatasets[i]

          databases = databases.concat(dataset.databases)

          try {
            const tics = dataset.diagnostics.filter((diagnostic : any) => diagnostic.type === 'TIC')
            const tic = tics[0].images.filter((image: any) => image.key === 'TIC' && image.format === 'NPY')
            const { data, shape } = await readNpy(tic[0].url)
            const metadata = safeJsonParse(tics[0].data)
            metadata.maxTic = metadata.max_tic
            metadata.minTic = metadata.min_tic
            delete metadata.max_tic
            delete metadata.min_tic

            normalizationData[dataset.id] = {
              data,
              shape,
              metadata: metadata,
              type: 'TIC',
              showFullTIC: false,
              error: false,
            }
          } catch (e) {
            normalizationData[dataset.id] = {
              data: null,
              shape: null,
              metadata: null,
              showFullTIC: null,
              type: 'TIC',
              error: true,
            }
          }
        }

        state.databaseOptions = { database: uniqBy(databases, 'id') }
      }
      state.normalizationData = normalizationData
    })

    const loadAnnotations = () => { annotationQueryOptions.enabled = true }
    const datasets = computed(() => {
      if (datasetsResult.value) {
        return datasetsResult.value.allDatasets
      }
      return null
    })

    const requestAnnotations = async() => {
      state.isLoading = true
      loadAnnotations()
      state.isLoading = false
    }

    onMounted(() => {
      state.refsLoaded = true

      // add listener to query annotations again in case the filters change
      $store.watch((_, getters) => [getters.gqlAnnotationFilter,
        $store.getters.gqlDatasetFilter, $store.getters.gqlColocalizationFilter, $store.getters.ftsQuery],
      (filter, previousFilter) => {
        if (state.annotations && !isEqual(filter, previousFilter)) {
          state.offset = 0
          annotationQueryOptions.enabled = true
        }
      })
    })

    watchEffect(async() => {
      if (!state.grid && gridSettings.value) {
        const auxSettings = safeJsonParse(gridSettings.value.snapshot)
        state.nCols = auxSettings.nCols
        state.nRows = auxSettings.nRows
        state.grid = auxSettings.grid

        if (auxSettings.filter) {
          const filter = Object.assign({}, auxSettings.filter)
          $store.commit('updateFilter', filter)
        }

        if (auxSettings.colormap) {
          handleColormapChange(auxSettings.colormap)
        } else if ($route.query.cmap) {
          handleColormapChange($route.query.cmap)
        }

        if (auxSettings.norm) {
          handleNormalizationChange(true)
        } else if ($route.query.norm) {
          handleNormalizationChange(true)
        }

        if (auxSettings.scaleType) {
          handleScaleTypeChange(auxSettings.scaleType)
        } else if ($route.query.scale) {
          handleScaleTypeChange($route.query.scale)
        }

        handleScaleBarColorChange(auxSettings.scaleBarColor)
        if (auxSettings.lockedIntensityTemplate) {
          handleTemplateChange(auxSettings.lockedIntensityTemplate)
        } else if (
          Array.isArray(auxSettings.globalLockedIntensities)
          && auxSettings.globalLockedIntensities.length === 2) {
          const intensities : [number | undefined, number | undefined] = [
            auxSettings.globalLockedIntensities[0] === null ? undefined : auxSettings.globalLockedIntensities[0],
            auxSettings.globalLockedIntensities[1] === null ? undefined : auxSettings.globalLockedIntensities[1],
          ]
          handleIntensitiesChange(intensities)
        }

        await requestAnnotations()

        // sets lock template after grid mounted
        if ($store.getters.settings.annotationView.lockTemplate) {
          handleTemplateChange($store.getters.settings.annotationView.lockTemplate)
        }
      }
    })

    const resetViewPort = (event: any) => {
      if (event) {
        event.stopPropagation()
      }
      state.globalImageSettings.resetViewPort = !state.globalImageSettings.resetViewPort
    }

    const handleScaleBarColorChange = (scaleBarColor: string) => {
      state.globalImageSettings.scaleBarColor = scaleBarColor
    }

    const handleScaleTypeChange = (scaleType: string) => {
      state.globalImageSettings.scaleType = scaleType
    }

    const handleColormapChange = (colormap: string) => {
      state.globalImageSettings.colormap = colormap
    }

    const handleNormalizationChange = (isNormalized: boolean) => {
      state.globalImageSettings.isNormalized = isNormalized
    }

    const handleTemplateChange = (dsId: string) => {
      state.globalImageSettings.selectedLockTemplate = dsId
      $store.commit('setLockTemplate', dsId)
    }

    const handleIntensitiesChange = (intensities: [number | undefined, number | undefined]) => {
      state.globalImageSettings.globalLockedIntensities = intensities
    }

    const handleRowChange = (idx: number) => {
      if (idx !== -1) {
        state.isLoading = true
        state.selectedAnnotation = idx
        setTimeout(() => {
          state.isLoading = false
        }, 500)
      }
    }

    const renderInfo = () => {
      const nCols = state.nCols
      const nRows = state.nRows

      if (
        state.selectedAnnotation === undefined
        || state.selectedAnnotation === -1
        || !state.annotations[state.selectedAnnotation]) {
        return <div class='ds-comparison-info'/>
      }

      const selectedAnnotation = state.annotations[state.selectedAnnotation].annotations[0]
      let possibleCompounds : any = []
      let isomers : any = []
      let isobars : any = []

      state.annotations[state.selectedAnnotation].annotations.forEach((annotation: any) => {
        possibleCompounds = possibleCompounds.concat(annotation.possibleCompounds)
        isomers = isomers.concat(annotation.isomers)
        isobars = isobars.concat(annotation.isobars)
      })

      // @ts-ignore TS2604
      const candidateMolecules = () => <CandidateMoleculesPopover
        placement="bottom"
        possibleCompounds={possibleCompounds}
        isomers={isomers}
        isobars={isobars}>
        <MolecularFormula
          class="sf-big text-2xl"
          ion={selectedAnnotation.ion}
        />
      </CandidateMoleculesPopover>

      return (
        <div class='ds-comparison-info'>
          {candidateMolecules()}
          <CopyButton
            class="ml-1"
            text={selectedAnnotation.ion}>
            Copy ion to clipboard
          </CopyButton>
          <span class="text-2xl flex items-baseline ml-4">
            { selectedAnnotation.mz.toFixed(4) }
            <span class="ml-1 text-gray-700 text-sm">m/z</span>
            <CopyButton
              class="self-start"
              text={selectedAnnotation.mz.toFixed(4)}>
              Copy m/z to clipboard
            </CopyButton>
          </span>
          <DatasetComparisonShareLink
            viewId={snapshotId}
            nCols={nCols}
            nRows={nRows}
            lockedIntensityTemplate={state.globalImageSettings.selectedLockTemplate}
            globalLockedIntensities={state.globalImageSettings.globalLockedIntensities}
            scaleBarColor={state.globalImageSettings.scaleBarColor}
            scaleType={state.globalImageSettings.scaleType}
            colormap={state.globalImageSettings.colormap}
            settings={gridSettings.value?.snapshot}
            selectedAnnotation={state.selectedAnnotation}
            sourceDsId={sourceDsId}
            name={$route.name}
            params={$route.params}
            query={$route.query}/>
        </div>
      )
    }

    const renderImageGallery = (nCols: number, nRows: number) => {
      return (
        <CollapseItem
          id="annot-img-collapse"
          name="images"
          class="ds-collapse el-collapse-item--no-padding relative">
          <MainImageHeader
            class='dataset-comparison-item-header dom-to-image-hidden'
            slot="title"
            isActive={false}
            scaleBarColor={state.globalImageSettings?.scaleBarColor}
            onScaleBarColorChange={handleScaleBarColorChange}
            scaleType={state.globalImageSettings?.scaleType}
            onScaleTypeChange={handleScaleTypeChange}
            showIntensityTemplate={true}
            showNormalizedBadge={state.collapse.includes('images') && state.globalImageSettings.isNormalized}
            onNormalizationChange={handleNormalizationChange}
            colormap={state.globalImageSettings?.colormap}
            onColormapChange={handleColormapChange}
            lockedTemplate={state.globalImageSettings.selectedLockTemplate}
            onTemplateChange={handleTemplateChange}
            showOpticalImage={false}
            hasOpticalImage={false}
            resetViewport={resetViewPort}
            toggleOpticalImage={() => {}}
            lockTemplateOptions={uniqBy(datasets.value, 'id')}
          />
          <ImageSaver
            class="absolute top-0 right-0 mt-2 mr-2 dom-to-image-hidden"
            domNode={gridNode.value}
          />
          <div class='dataset-comparison-grid' ref={gridNode}>
            {
              state.collapse.includes('images')
              && <DatasetComparisonGrid
                ref={imageGrid}
                nCols={nCols}
                nRows={nRows}
                resetViewPort={state.globalImageSettings.resetViewPort}
                onResetViewPort={resetViewPort}
                onLockAllIntensities={handleTemplateChange}
                onIntensitiesChange={handleIntensitiesChange}
                lockedIntensityTemplate={state.globalImageSettings.selectedLockTemplate}
                globalLockedIntensities={state.globalImageSettings.globalLockedIntensities}
                scaleBarColor={state.globalImageSettings.scaleBarColor}
                scaleType={state.globalImageSettings.scaleType}
                colormap={state.globalImageSettings.colormap}
                isNormalized={state.globalImageSettings.isNormalized}
                settings={gridSettings}
                annotations={state.annotations || []}
                normalizationData={state.normalizationData}
                datasets={datasets.value || []}
                selectedAnnotation={state.selectedAnnotation}
                isLoading={state.isLoading || annotationsLoading.value}
              />
            }
          </div>
        </CollapseItem>)
    }

    const renderCompounds = () => {
      const annotations = state.selectedAnnotation >= 0 && state.annotations[state.selectedAnnotation]
        ? state.annotations[state.selectedAnnotation].annotations : []

      // @ts-ignore TS2604
      const relatedMolecules = () => <RelatedMolecules
        query="isomers"
        annotation={annotations[0]}
        annotations={annotations}
        databaseId={$store.getters.filter.database || 1}
        hideFdr
      />

      return (<CollapseItem
        id="annot-img-collapse"
        name="compounds"
        title="Molecules"
        class="ds-collapse el-collapse-item--no-padding relative">
        {
          !state.isLoading
          && state.collapse.includes('compounds')
          && (Array.isArray(annotations) && annotations.length > 0)
          && relatedMolecules()
        }
        {
          !state.isLoading
          && state.collapse.includes('compounds')
          && (!Array.isArray(annotations) || annotations.length < 1)
          && <div class='flex w-full items-center justify-center'>No data</div>
        }
      </CollapseItem>)
    }

    return () => {
      const nCols = state.nCols
      const nRows = state.nRows

      if (!snapshotId) {
        return (
          <div class='dataset-comparison-page w-full flex flex-wrap flex-row items-center justify-center'>
            Not found
          </div>)
      }
      return (
        <div class='dataset-comparison-page w-full flex flex-wrap flex-row'>
          <FilterPanel
            class='w-full'
            level='annotation'
            hiddenFilters={['datasetIds']}
            fixedOptions={state.databaseOptions}
          />
          <div class='dataset-comparison-wrapper w-full md:w-6/12 relative'>
            {
              state.annotations
              && <DatasetComparisonAnnotationTable
                isLoading={state.annotationLoading}
                annotations={state.annotations.map((ion: any) => {
                  return {
                    ...ion.annotations[0],
                    msmScore: Math.max(...ion.annotations.map((annot: any) => annot.msmScore)),
                    fdrlevel: Math.min(...ion.annotations.map((annot: any) => annot.fdrlevel)),
                    datasetcount: (ion.datasetIds || []).length,
                    rawAnnotations: ion.annotations,
                  }
                })}
                onRowChange={handleRowChange}
              />
            }
            {
              annotationsLoading.value
              && <div class='w-full absolute text-center top-0'>
                <i
                  class="el-icon-loading"
                />
              </div>
            }
          </div>
          <div class='dataset-comparison-wrapper  w-full  md:w-6/12'>
            <Collapse
              value={state.collapse}
              id="annot-content"
              class="border-0"
              onChange={(activeNames: string[]) => {
                state.collapse = activeNames
              }}>
              {renderInfo()}
              {renderImageGallery(nCols, nRows)}
              {renderCompounds()}
            </Collapse>
          </div>
        </div>
      )
    }
  },
})
