import { computed, defineComponent, onMounted, onUnmounted, reactive, ref, watch } from '@vue/composition-api'
// @ts-ignore
import ECharts from 'vue-echarts'
import { use } from 'echarts/core'
import {
  CanvasRenderer,
} from 'echarts/renderers'
import {
  BarChart,
  ScatterChart,
  LineChart,
} from 'echarts/charts'
import {
  GridComponent,
  TooltipComponent,
  ToolboxComponent,
  LegendComponent,
  DataZoomComponent,
  MarkPointComponent,
  TitleComponent,
  VisualMapPiecewiseComponent,
  VisualMapContinuousComponent,
  MarkLineComponent,
} from 'echarts/components'
import './DashboardScatterChart.scss'

use([
  CanvasRenderer,
  BarChart,
  ScatterChart,
  LineChart,
  GridComponent,
  TooltipComponent,
  ToolboxComponent,
  LegendComponent,
  DataZoomComponent,
  MarkPointComponent,
  TitleComponent,
  VisualMapPiecewiseComponent,
  MarkLineComponent,
  VisualMapContinuousComponent,
])

interface DashboardScatterChartProps {
  isEmpty: boolean
  isLoading: boolean
  isDataLoading: boolean
  data: any[]
  visualMap: any
  xAxis: any[]
  yAxis: any[]
  annotatedData: any[]
  peakFilter: number
  size: number
  xOption: string
  yOption: string
}

interface DashboardScatterChartState {
  scaleIntensity: boolean
  chartOptions: any
  size: number
}

const PEAK_FILTER = {
  ALL: 1,
  FDR: 2,
}

export const DashboardScatterChart = defineComponent<DashboardScatterChartProps>({
  name: 'DashboardScatterChart',
  props: {
    isEmpty: {
      type: Boolean,
      default: true,
    },
    isLoading: {
      type: Boolean,
      default: false,
    },
    isDataLoading: {
      type: Boolean,
      default: false,
    },
    xAxis: {
      type: Array,
      default: () => [],
    },
    yAxis: {
      type: Array,
      default: () => [],
    },
    data: {
      type: Array,
      default: () => [],
    },
    visualMap: {
      type: Object,
      default: {},
    },
    annotatedData: {
      type: Array,
      default: () => [],
    },
    peakFilter: {
      type: Number,
      default: PEAK_FILTER.ALL,
    },
    size: {
      type: Number,
      default: 600,
    },
    xOption: {
      type: String,
    },
    yOption: {
      type: String,
    },
  },
  setup(props, { emit }) {
    const spectrumChart = ref(null)
    const xAxisData = computed(() => props.xAxis)
    const yAxisData = computed(() => props.yAxis)

    const state = reactive<DashboardScatterChartState>({
      scaleIntensity: false,
      size: 600,
      chartOptions: {
        title: {
          text: '',
        },
        toolbox: {
          feature: {
            saveAsImage: {
              title: ' ',
            },
          },
        },
        tooltip: {
          position: 'top',
          formatter: function(params: any) {
            return 'Fraction detected: ' + (params.value[4] || 0).toFixed(2) + ' '
              + params.data?.label?.y + ' in ' + params.data?.label?.x
          },
        },
        grid: {
          left: '5%',
          top: 20,
          right: '5%',
          bottom: 60,
          containLabel: true,
        },
        xAxis: {
          type: 'category',
          data: [],
          boundaryGap: true,
          splitLine: {
            show: false,
          },
          axisLine: {
            show: false,
          },
          axisLabel: {
            show: true,
            interval: 0,
            rotate: 30,
          },
          position: 'top',
        },
        yAxis: {
          type: 'category',
          data: [],
          axisLine: {
            show: false,
          },
          axisLabel: {
            show: true,
            interval: 0,
            height: 30,
          },
        },
        series: [{
          type: 'scatter',
          markLine: {},
          symbolSize: function(val: any) {
            return val[2] * 2
          },
          itemStyle: {
            borderColor: 'black',
          },
          data: [],
        }],
      },
    })

    const chartData = computed(() => props.data)
    const visualMap = computed(() => props.visualMap)
    const chartOptions = computed(() => {
      if (!xAxisData.value || !chartData.value || !visualMap.value) {
        return state.chartOptions
      }

      const auxOptions = state.chartOptions
      const globalCategories : any = {}
      const markData : any = []
      yAxisData.value.forEach((label: string, idx: number) => {
        const re = /(.+)\s-agg-\s(.+)/
        const found = label.match(re)
        const cat = label.replace(re, '$1')
        if (found) {
          globalCategories[cat] = idx
        }
      })
      Object.keys(globalCategories).map((key: string) => {
        markData.push({
          name: key,
          yAxis: globalCategories[key],
          label: {
            formatter: key,
            position: 'end',
            width: 100,
            overflow: 'break',
          },
          lineStyle: {
            color: 'transparent',
          },
        })
      })

      if (props.yOption === 'fine_class' || props.yOption === 'fine_path') {
        auxOptions.grid.right = 100
      } else {
        auxOptions.grid.right = '5%'
      }

      auxOptions.xAxis.data = xAxisData.value
      auxOptions.yAxis.data = yAxisData.value
        .map((label: string) => label.replace(/.+-agg-\s(.+)/, '$1'))
      auxOptions.series[0].data = chartData.value
      auxOptions.series[0].markLine.data = markData
      if (visualMap.value && visualMap.value.type) {
        auxOptions.visualMap = visualMap.value
      }
      return auxOptions
    })

    const handleChartResize = () => {
      const chartRef : any = spectrumChart.value
      if (chartRef && chartRef.chart) {
        chartRef.chart.resize()
      }
    }

    onMounted(() => {
      window.addEventListener('resize', handleChartResize)
    })

    onUnmounted(() => {
      window.removeEventListener('resize', handleChartResize)
    })

    // set images and annotation related items when selected annotation changes
    watch(() => props.size, async(newValue) => {
      state.size = props.size < 600 ? 600 : props.size
      setTimeout(() => handleChartResize(), 500)
    })

    const handleZoomReset = () => {
      if (spectrumChart && spectrumChart.value) {
        // @ts-ignore
        spectrumChart.value.chart.dispatchAction({
          type: 'dataZoom',
          start: 0,
          end: 100,
        })
      }
    }

    const handleItemSelect = (item: any) => {
      if (item.targetType === 'axisName') {
        state.scaleIntensity = !state.scaleIntensity
      } else {
        emit('itemSelected', item)
      }
    }

    const renderSpectrum = () => {
      const { isLoading, isDataLoading } = props

      return (
        <div class='chart-holder'
          style={{ height: `${state.size}px` }}>
          {
            (isLoading || isDataLoading)
            && <div class='loader-holder'>
              <div>
                <i
                  class="el-icon-loading"
                />
              </div>
            </div>
          }
          <ECharts
            ref={spectrumChart}
            autoResize={true}
            {...{
              on: {
                'zr:dblclick': handleZoomReset,
                click: handleItemSelect,
              },
            }}
            class='chart'
            style={{ height: `${state.size}px` }}
            options={chartOptions.value}/>
        </div>
      )
    }

    return () => {
      return (
        <div class={'dataset-browser-spectrum-container'}>
          {renderSpectrum()}
        </div>
      )
    }
  },
})
