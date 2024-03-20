import { defineComponent, computed } from 'vue'
import VisibilityBadge from '../common/VisibilityBadge'
import ElapsedTime from '../../../components/ElapsedTime'
import { plural } from '../../../lib/vueFilters'
import { DatasetDetailItem } from '../../../api/dataset'
import { capitalize, get } from 'lodash-es'
import FilterLink from './FilterLink'
import { formatDatabaseLabel } from '../../MolecularDatabases/formatting'
import CopyButton from '../../../components/CopyButton.vue'
import { useStore } from 'vuex'
import { useRouter } from 'vue-router'
import { ElDropdown, ElDropdownMenu, ElDropdownItem } from '../../../lib/element-plus'

type FilterField = keyof DatasetDetailItem | 'analyzerType'

const DatasetItemMetadata = defineComponent({
  name: 'DatasetItemMetadata',
  components: {
    ElDropdown,
    ElDropdownMenu,
    ElDropdownItem,
  },
  props: {
    dataset: { type: Object as () => DatasetDetailItem, required: true },
    metadata: { type: Object as () => any, required: true },
    hideGroupMenu: { type: Boolean, default: false },
  },
  setup(props, ctx) {
    const store = useStore()
    const router = useRouter()

    const databaseLabel = computed(() =>
      formatDatabaseLabel({
        name: props.dataset.fdrCounts.dbName,
        version: props.dataset.fdrCounts.dbVersion,
      })
    )

    const addFilter = (field: FilterField) => {
      const filter = Object.assign({}, store.getters.filter)
      if (field === 'polarity') {
        filter.polarity = capitalize(props.dataset.polarity)
      } else if (field === 'submitter') {
        filter[field] = props.dataset.submitter.id
      } else if (field === 'group') {
        filter[field] = props.dataset.group.id
      } else if (field === 'analyzerType') {
        filter[field] = props.dataset.analyzer.type
      } else {
        filter[field] = props.dataset[field]
      }
      store.commit('updateFilter', filter)
      ctx.emit('filterUpdate', filter)
    }

    const handleDropdownCommand = (command: string) => {
      if (command === 'filter_group') {
        addFilter('group')
      } else if (command === 'view_group') {
        router.push({
          name: 'group',
          params: {
            groupIdOrSlug: props.dataset.group.id,
          },
        })
      }
    }

    return () => {
      const { dataset, metadata, hideGroupMenu } = props
      const { mz: rpAtMz = 0, Resolving_Power: rp = 0 } =
        get(metadata, ['MS_Analysis', 'Detector_Resolving_Power']) || {}

      const filterableItem = (field: FilterField, name: string, text: string) => (
        <span class="ds-add-filter" title={`Filter by ${name}`} onClick={() => addFilter(field)}>
          {text}
        </span>
      )

      return (
        <div class="ds-info">
          <div class="ds-item-line flex">
            <span title={dataset.name} class="font-bold truncate">
              {dataset.name}
            </span>
            {!dataset.isPublic && <VisibilityBadge datasetId={dataset.id} />}
            <CopyButton class="ml-1" isId text={dataset.id}>
              Copy dataset id to clipboard
            </CopyButton>
          </div>
          <div class="ds-item-line text-gray-700">
            {filterableItem('organism', 'species', dataset.organism || '')}
            {', '}
            {filterableItem('organismPart', 'organism part', (dataset.organismPart || '').toLowerCase())}{' '}
            {filterableItem('condition', 'condition', `(${(dataset.condition || '').toLowerCase()})`)}
          </div>
          <div class="ds-item-line">
            {filterableItem('ionisationSource', 'ionisation source', dataset.ionisationSource)}
            {' + '}
            {filterableItem('analyzerType', 'analyzer type', dataset.analyzer?.type)}
            {', '}
            {filterableItem('polarity', 'polarity', `${dataset.polarity?.toLowerCase()} mode`)}
            {', RP '}
            {(rp / 1000).toFixed(0)}
            {'k @ '}
            {rpAtMz}
          </div>

          <div class="ds-item-line">
            Submitted <ElapsedTime date={dataset.uploadDT} />
            {' by '}
            {filterableItem('submitter', 'submitter', dataset.submitter?.name)}
            {dataset.group && (
              <span>
                {', '}
                <ElDropdown
                  showTimeout={50}
                  placement="bottom"
                  disabled={hideGroupMenu}
                  trigger={'hover'}
                  onCommand={handleDropdownCommand}
                  v-slots={{
                    dropdown: () => (
                      <ElDropdownMenu>
                        <ElDropdownItem command="filter_group">Filter by this group</ElDropdownItem>
                        <ElDropdownItem command="view_group">View group</ElDropdownItem>
                      </ElDropdownMenu>
                    ),
                    default: () => (
                      <span class="text-base text-primary cursor-pointer" onClick={() => addFilter('group')}>
                        {dataset.group?.shortName}
                      </span>
                    ),
                  }}
                />
              </span>
            )}
          </div>
          {dataset.status === 'FINISHED' && dataset.fdrCounts && (
            <div class="ds-item-line">
              <span>
                <FilterLink filter={{ database: dataset.fdrCounts.databaseId, datasetIds: [dataset.id] }}>
                  {plural(dataset.fdrCounts.counts.join(', '), 'annotation', 'annotations')}
                </FilterLink>
                {' @ FDR '}
                {dataset.fdrCounts.levels.join(', ')}% ({databaseLabel.value})
              </span>
            </div>
          )}
        </div>
      )
    }
  },
})
export default DatasetItemMetadata
