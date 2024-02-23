import { defineComponent } from 'vue'
import { ElTable, ElTableColumn, ElButton } from '../../../lib/element-plus'
import { DatasetAnnotationCount } from '../../../api/dataset'
import { encodeParams } from '../../Filters'
import { useRouter } from 'vue-router'
import RouterLink from '../../../components/RouterLink'

interface AnnotationCountTableData {
  name: string
  [key: string]: any // dynamic keys according to fdrs
}

interface AnnotationCountTableProps {
  id: string
  sumRowLabel: string
  btnLabel: string
  title: string
  data: DatasetAnnotationCount[]
  header: string[]
  headerTitleSuffix: string
}

export const AnnotationCountTable = defineComponent({
  name: 'AnnotationCountTable',
  props: {
    id: { type: String, default: '' },
    title: { type: String, default: '# of annotations with FDR% <=' },
    sumRowLabel: { type: String, default: 'Total Annotations' },
    btnLabel: { type: String, default: 'Browse annotations' },
    data: { type: Array, default: () => [] },
    header: { type: Array, default: () => [5, 10, 20, 50] },
    headerTitleSuffix: { type: String, default: '%' },
  },
  setup(props: AnnotationCountTableProps) {
    const router = useRouter()
    const annotationsLink = (datasetId: string, database?: string, fdrLevel?: number) => {
      const query = {
        database,
        fdrLevel,
        datasetIds: [datasetId],
      }
      // delete undefined so that filter do not replace the nulls and make the navigation back weird
      if (!database) {
        delete query.database
      }
      if (!fdrLevel) {
        delete query.fdrLevel
      }

      return {
        name: 'dataset-annotations',
        params: { dataset_id: datasetId },
        query: encodeParams(query),
      }
    }

    const mountTableData = (data: DatasetAnnotationCount[], header: string[]): AnnotationCountTableData[] => {
      if (!data) {
        return []
      }

      return data.map((item: DatasetAnnotationCount) => {
        const obj = {} as AnnotationCountTableData
        obj.id = item?.databaseId
        obj.name = `${item?.dbName}${item?.dbVersion ? `-${item?.dbVersion}` : ''}`
        header.forEach((headerEl: string) => {
          obj[headerEl] = Array.isArray(item?.counts)
            ? ((item.counts.find((count: any) => count.level === headerEl) || {}) as any).n
            : '-'
        })
        return obj
      })
    }

    const browseAnnotations = () => {
      const { id } = props
      router.push(annotationsLink(id))
    }

    const formatTitle = (col: number | string) => {
      const { headerTitleSuffix } = props
      return `${col === 'name' ? 'Database' : col}${col !== 'name' ? headerTitleSuffix : ''}`
    }

    const formatCell = (row: any, column: any, cellValue: any, rowIndex: number, colIndex: number) => {
      const { id } = props
      if (colIndex === 0) {
        return <RouterLink to={annotationsLink(id?.toString(), row?.id, undefined)}>{cellValue}</RouterLink>
      }

      return (
        <RouterLink to={annotationsLink(id?.toString(), row?.id, parseInt(column?.property, 10) / 100)}>
          {cellValue}
        </RouterLink>
      )
    }

    const getSummaries = (param: any) => {
      const { columns, data } = param
      return columns.map((col: any, colIndex: number) => {
        if (colIndex === 0) {
          return props.sumRowLabel
        }
        const currentFDR = props.header[colIndex - 1]

        const reducer = (accumulator: number, currentValue: any) => {
          return accumulator + currentValue[currentFDR]
        }

        return (
          <RouterLink
            key={colIndex}
            to={annotationsLink(props.id?.toString(), undefined, parseInt(currentFDR, 10) / 100)}
          >
            {data.reduce(reducer, 0)}
          </RouterLink>
        )
      })
    }

    return () => {
      const { data, header, sumRowLabel, btnLabel, title } = props
      const tableData = mountTableData(data, header)
      return (
        <div class="relative text-center">
          <h4>{title}</h4>
          <ElTable
            data={tableData}
            show-summary={true}
            sum-text={sumRowLabel}
            summary-method={getSummaries}
            style={{ width: '100%', marginTop: 20 }}
          >
            {['name'].concat(header).map((col, colIndex) => {
              return (
                <ElTableColumn
                  sortable={true}
                  key={colIndex}
                  prop={col.toString()}
                  formatter={(row: any, column: any, cellValue: any, index: number) =>
                    formatCell(row, column, cellValue, index, colIndex)
                  }
                  label={formatTitle(col)}
                />
              )
            })}
          </ElTable>
          <div class="text-right mt-2">
            <ElButton onClick={browseAnnotations}>{btnLabel}</ElButton>
          </div>
        </div>
      )
    }
  },
})
