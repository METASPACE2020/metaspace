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
  HeatmapChart,
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
} from 'echarts/components'
import './DashboardHeatmapChart.scss'

use([
  CanvasRenderer,
  BarChart,
  ScatterChart,
  LineChart,
  HeatmapChart,
  GridComponent,
  TooltipComponent,
  ToolboxComponent,
  LegendComponent,
  DataZoomComponent,
  MarkPointComponent,
  TitleComponent,
  VisualMapPiecewiseComponent,
  VisualMapContinuousComponent,
])

interface DashboardHeatmapChartProps {
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
}

interface DashboardHeatmapChartState {
  scaleIntensity: boolean
  chartOptions: any
  size: number
}

const PEAK_FILTER = {
  ALL: 1,
  FDR: 2,
}

// const items = ['not detected', 'M+H', 'M+Na', 'M+']

export const DashboardHeatmapChart = defineComponent<DashboardHeatmapChartProps>({
  name: 'DashboardHeatmapChart',
  props: {
    isEmpty: {
      type: Boolean,
      default: true,
    },
    isLoading: {
      type: Boolean,
      default: false,
    },
    size: {
      type: Number,
      default: 600,
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
  },
  setup(props, { emit }) {
    const spectrumChart = ref(null)
    const xAxisData = computed(() => props.xAxis)
    const yAxisData = computed(() => props.yAxis)

    const state = reactive<DashboardHeatmapChartState>({
      scaleIntensity: false,
      size: 600,
      chartOptions: {
        tooltip: {
          position: 'top',
          formatter: function(params: any) {
            return params.value[4].toFixed(2) + ' ' + params.data?.label?.y + ' in ' + params.data?.label?.x
          },
        },
        grid: {
          left: 2,
          top: 20,
          right: 20,
          bottom: 40,
          containLabel: true,
        },
        xAxis: {
          type: 'category',
          data: [],
          splitArea: {
            show: true,
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
          splitArea: {
            show: true,
          },
          axisLabel: {
            show: true,
            interval: 0,
            height: 30,
          },
        },
        visualMap: {
          min: 0,
          max: 10,
          calculable: true,
          orient: 'horizontal',
          left: 'center',
          bottom: '15%',
        },
        toolbox: {
          feature: {
            saveAsImage: {
              title: ' ',
            },
          },
        },
        series: [{
          name: 'Punch Card',
          type: 'heatmap',
          data: [],
          label: {
            normal: {
              show: true,
              formatter: (param: any) => {
                return param.data?.label?.molecule ? '' : 'Not detected'
              },
            },
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)',
            },
          },
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
      auxOptions.xAxis.data = xAxisData.value
      auxOptions.yAxis.data = yAxisData.value
      auxOptions.series[0].data = chartData.value
      if (visualMap.value && visualMap.value.type) {
        auxOptions.visualMap = visualMap.value
      }
      return state.chartOptions
    })

    const handleChartResize = () => {
      if (spectrumChart && spectrumChart.value) {
        // @ts-ignore
        spectrumChart.value.chart.resize()
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
        emit('itemSelected', item.data.mz)
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
