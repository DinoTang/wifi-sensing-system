"""
Wi-Fi Sensing System - Frontend Dashboard
Giao diện Web mô phỏng và giám sát tín hiệu CSI chuẩn bài báo khoa học PMC11479185:
- Nhận diện ranh giới sự kiện (Event Segmentation: Start & End)
- Dạng sóng CSI cao tần (Walking, Hand swing, Falling, Sitting down)
- Màn hình phát hiện có người trong phòng
"""

import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from nicegui import ui, app
from csi_source import CSIDataSource

# Khởi tạo nguồn dữ liệu
data_source = CSIDataSource(num_subcarriers=64)

# Dữ liệu ban đầu cho biểu đồ mẫu
initial_time, initial_sig, init_start, init_end = data_source.generate_benchmark_pattern("walking")

# Thiết lập giao diện
ui.query('.q-page').classes('bg-slate-50 dark:bg-slate-900')

# ==================== HEADER ====================
with ui.header().classes('bg-gradient-to-r from-slate-900 via-indigo-950 to-blue-950 text-white shadow-md p-3 px-6 flex justify-between items-center'):
    with ui.row().classes('items-center gap-3'):
        ui.icon('sensors', size='32px').classes('text-red-400')
        with ui.column().classes('gap-0'):
            ui.label('WI-FI CSI SENSING & ACTIVITY SEGMENTATION').classes('text-lg font-black tracking-wide')
            ui.label('Chuẩn bài báo nghiên cứu (PMC11479185 / Sensors 2024)').classes('text-xs text-indigo-300')

    with ui.row().classes('items-center gap-4'):
        with ui.badge(color='red-7').classes('px-3 py-1 text-xs font-mono'):
            ui.icon('timeline', size='14px').classes('mr-1')
            active_activity_badge = ui.label('Hành vi: (a) walking')

        with ui.badge(color='teal-7').classes('px-3 py-1 text-xs font-mono'):
            ui.icon('speed', size='14px').classes('mr-1')
            ui.label('Sampling: 100 Hz')

        dark = ui.dark_mode()
        ui.button(icon='dark_mode', on_click=dark.toggle).props('flat round dense text-color=white')

# ==================== NAVIGATION TABS ====================
with ui.tabs().classes('w-full bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 shadow-sm') as tabs:
    tab1 = ui.tab('Màn hình 1: Dạng sóng CSI & Phân đoạn Start - End', icon='show_chart').classes('font-bold text-sm')
    tab2 = ui.tab('Màn hình 2: Trạng thái phát hiện người', icon='sensor_occupied').classes('font-bold text-sm')

# ==================== TAB PANELS ====================
with ui.tab_panels(tabs, value=tab1).classes('w-full p-4 bg-transparent'):

    # -------------------------------------------------------------
    # MÀN HÌNH 1: DẠNG SÓNG CSI PHÂN ĐOẠN START - END (PMC11479185)
    # -------------------------------------------------------------
    with ui.tab_panel(tab1).classes('p-0 gap-4 flex flex-col'):
        
        # Thanh chọn chế độ phân tích 4 hành vi khoa học
        with ui.card().classes('w-full p-4 border border-slate-200 dark:border-slate-700 shadow-sm flex flex-col md:flex-row justify-between items-center gap-4'):
            with ui.column().classes('gap-0'):
                ui.label('Chọn mẫu hành vi từ nghiên cứu thực nghiệm (CSI Amplitude Variations):').classes('font-bold text-sm text-slate-800 dark:text-slate-100')
                ui.label('Trục X: Thời gian (ms) | Trục Y: Normalized CSI Amplitude | Vạch đỏ: Điểm bắt đầu (start) và kết thúc (end)').classes('text-xs text-slate-500')

            with ui.row().classes('gap-2'):
                btn_walk = ui.button('(a) Walking', icon='directions_walk').props('dense color=primary')
                btn_hand = ui.button('(b) Hand swing', icon='front_hand').props('dense outline color=primary')
                btn_fall = ui.button('(c) Falling', icon='personal_injury').props('dense outline color=negative')
                btn_sit = ui.button('(d) Sitting down', icon='chair').props('dense outline color=primary')
                btn_live = ui.button('📡 Real-time Stream', icon='stream').props('dense outline color=positive')

        # BIỂU ĐỒ CHÍNH CHUẨN PMC11479185
        with ui.card().classes('w-full p-4 border border-slate-200 dark:border-slate-700 shadow-md'):
            with ui.row().classes('w-full justify-between items-center mb-1'):
                chart_title = ui.label('(a) walking - Phân tích biên độ CSI và phát hiện ranh giới chuyển động').classes('font-mono font-bold text-base text-slate-900 dark:text-slate-100')
                ui.badge('Start - End Detected', color='red-6').classes('text-xs font-mono')

            # Cấu hình Apache ECharts tái hiện chính xác hình ảnh người dùng cung cấp
            def build_chart_options(time_data, signal_data, start_ms, end_ms, y_min=-0.025, y_max=0.03):
                mark_lines = []
                mark_areas = []

                if start_ms is not None and end_ms is not None and start_ms > 0:
                    mark_lines = [
                        {
                            'xAxis': start_ms,
                            'label': {
                                'formatter': 'start',
                                'position': 'insideStartTop',
                                'color': '#dc2626',
                                'fontWeight': 'bold',
                                'fontSize': 14,
                                'distance': 10
                            },
                            'lineStyle': {'color': '#dc2626', 'width': 2, 'type': 'dashed'}
                        },
                        {
                            'xAxis': end_ms,
                            'label': {
                                'formatter': 'end',
                                'position': 'insideStartTop',
                                'color': '#dc2626',
                                'fontWeight': 'bold',
                                'fontSize': 14,
                                'distance': 10
                            },
                            'lineStyle': {'color': '#dc2626', 'width': 2, 'type': 'dashed'}
                        }
                    ]
                    mark_areas = [[
                        {
                            'xAxis': start_ms,
                            'itemStyle': {'color': 'rgba(239, 68, 68, 0.08)'}
                        },
                        {
                            'xAxis': end_ms
                        }
                    ]]

                return {
                    'grid': {'left': '5%', 'right': '4%', 'bottom': '12%', 'top': '10%', 'containLabel': True},
                    'tooltip': {
                        'trigger': 'axis',
                        'formatter': 'Thời gian: {b} ms<br/>Biên độ CSI: {c}'
                    },
                    'xAxis': {
                        'type': 'category',
                        'name': 'Time (milliseconds)',
                        'nameLocation': 'middle',
                        'nameGap': 30,
                        'nameTextStyle': {'fontSize': 13, 'fontWeight': 'bold'},
                        'data': time_data,
                        'axisLine': {'lineStyle': {'color': '#64748b'}},
                        'splitLine': {'show': False}
                    },
                    'yAxis': {
                        'type': 'value',
                        'name': 'Normalized Amplitude',
                        'nameLocation': 'middle',
                        'nameGap': 50,
                        'nameTextStyle': {'fontSize': 13, 'fontWeight': 'bold'},
                        'min': y_min,
                        'max': y_max,
                        'axisLine': {'show': True, 'lineStyle': {'color': '#64748b'}},
                        'splitLine': {'lineStyle': {'color': '#e2e8f0', 'type': 'dashed'}}
                    },
                    'series': [{
                        'name': 'CSI Signal',
                        'type': 'line',
                        'showSymbol': False,
                        'smooth': 0.1,  # đường sóng nét gai nhọn tự nhiên như hình
                        'data': signal_data,
                        'lineStyle': {'color': '#0f172a', 'width': 1.6},
                        'markLine': {
                            'symbol': ['none', 'none'],
                            'data': mark_lines
                        },
                        'markArea': {
                            'data': mark_areas
                        }
                    }],
                    'animation': False
                }

            start_t = initial_time[init_start] if init_start >= 0 else None
            end_t = initial_time[init_end] if init_end >= 0 else None
            chart_main = ui.echart(build_chart_options(initial_time, initial_sig, start_t, end_t, -0.025, 0.025)).classes('w-full h-96')

        # Hàng 4 thẻ so sánh nhanh đặc trưng các hành vi
        with ui.row().classes('w-full grid grid-cols-1 md:grid-cols-4 gap-4'):
            with ui.card().classes('p-3 border-l-4 border-blue-500 bg-white dark:bg-slate-800 shadow-sm'):
                ui.label('(a) Walking').classes('font-bold text-sm text-blue-600')
                ui.label('Dao động tần số cao liên tục do chuyển động bước chân & vung tay đồng thời.').classes('text-xs text-slate-500 mt-1')

            with ui.card().classes('p-3 border-l-4 border-indigo-500 bg-white dark:bg-slate-800 shadow-sm'):
                ui.label('(b) Hand Swing').classes('font-bold text-sm text-indigo-600')
                ui.label('Dạng đường bao vòm cong (envelope) với gợn sóng nhỏ của cánh tay phản xạ.').classes('text-xs text-slate-500 mt-1')

            with ui.card().classes('p-3 border-l-4 border-red-500 bg-white dark:bg-slate-800 shadow-sm'):
                ui.label('(c) Falling (Ngã)').classes('font-bold text-sm text-red-600')
                ui.label('Đột biến biên độ mạnh tức thì rồi sụt giảm sâu khi cơ thể rơi xuống sàn.').classes('text-xs text-slate-500 mt-1')

            with ui.card().classes('p-3 border-l-4 border-amber-500 bg-white dark:bg-slate-800 shadow-sm'):
                ui.label('(d) Sitting Down').classes('font-bold text-sm text-amber-600')
                ui.label('Xuất hiện pha lõm âm rõ rệt sau đó phục hồi về trạng thái cân bằng mới.').classes('text-xs text-slate-500 mt-1')

    # -------------------------------------------------------------
    # MÀN HÌNH 2: PHÁT HIỆN CÓ NGƯỜI
    # -------------------------------------------------------------
    with ui.tab_panel(tab2).classes('p-0 gap-4 flex flex-col'):
        with ui.row().classes('w-full grid grid-cols-1 lg:grid-cols-3 gap-4'):
            
            # Primary Detection Status Hero Card
            with ui.card().classes('lg:col-span-2 p-8 border border-slate-200 dark:border-slate-700 shadow-md flex flex-col justify-center items-center text-center') as status_card:
                status_icon_wrap = ui.element('div').classes('p-6 rounded-full bg-red-50 dark:bg-red-950/40 border-2 border-red-500 transition-all duration-300')
                with status_icon_wrap:
                    status_icon = ui.icon('directions_walk', size='96px').classes('text-red-500')

                status_title = ui.label('CÓ NGƯỜI (ĐANG DI CHUYỂN / WALKING)').classes('text-2xl font-black mt-4 text-red-600 dark:text-red-400 tracking-wider')
                status_subtext = ui.label('Tín hiệu CSI đang trong vùng hoạt động [START → END]').classes('text-sm text-slate-400 mt-1')

                # Confidence meter
                with ui.column().classes('w-full max-w-md mt-6 gap-2'):
                    with ui.row().classes('w-full justify-between text-xs font-semibold'):
                        ui.label('Độ tin cậy nhận diện (Confidence):')
                        conf_label = ui.label('98.0%').classes('font-bold text-red-500')
                    conf_progress = ui.linear_progress(value=0.98).props('rounded size=10px color=red')

            # Segmenter & Threshold Info Card
            with ui.card().classes('p-6 border border-slate-200 dark:border-slate-700 shadow-md flex flex-col justify-between'):
                with ui.column().classes('w-full gap-3'):
                    ui.label('Thuật toán phân đoạn sự kiện').classes('font-bold text-base border-b pb-2')

                    ui.label('Ngưỡng phát hiện Start / End:').classes('text-xs text-slate-500 font-semibold')
                    threshold_slider = ui.slider(min=0.002, max=0.02, step=0.001, value=0.008).props('label label-always color=red')
                    
                    with ui.row().classes('items-center justify-between text-xs text-slate-400'):
                        ui.label('Nhạy (Bắt sớm)')
                        ui.label('Chặt chẽ (Lọc nhiễu)')

                    ui.separator().classes('my-2')

                    with ui.column().classes('gap-1'):
                        ui.label('Trạng thái vùng tín hiệu:').classes('text-xs text-slate-500 font-semibold')
                        region_badge = ui.badge('NẰM TRONG VÙNG SỰ KIỆN', color='red-7').classes('text-xs py-1 px-2')

                    ui.button('Hiệu chuẩn nền phòng trống', icon='tune', on_click=data_source.calibrate_baseline if hasattr(data_source, 'calibrate_baseline') else lambda: None).props('outline color=primary class="w-full mt-2"')
                    ui.label('Thiết lập baseline khi phòng hoàn toàn không có người.').classes('text-[11px] text-slate-400 italic text-center')

        # Activity Timeline / Event Log
        with ui.card().classes('w-full p-6 border border-slate-200 dark:border-slate-700 shadow-sm'):
            ui.label('Nhật ký sự kiện ranh giới (Start / End Event Log)').classes('font-bold text-base mb-3')
            
            table_columns = [
                {'name': 'time', 'label': 'Thời gian', 'field': 'time', 'align': 'left'},
                {'name': 'state', 'label': 'Mốc sự kiện', 'field': 'state', 'align': 'left'},
                {'name': 'event', 'label': 'Chi tiết hành vi / Ranh giới', 'field': 'event', 'align': 'left'},
                {'name': 'confidence', 'label': 'Độ tin cậy', 'field': 'confidence', 'align': 'left'},
            ]
            log_table = ui.table(columns=table_columns, rows=data_source.event_logs).classes('w-full')

# ==================== LOGIC XỬ LÝ CHUYỂN MẪU HÀNH VI ====================
current_mode = "benchmark"  # 'benchmark' hoặc 'realtime'

def switch_pattern(pattern_name: str, y_min: float, y_max: float, label_text: str):
    global current_mode
    current_mode = "benchmark"
    data_source.current_activity = pattern_name
    active_activity_badge.set_text(f'Hành vi: {label_text}')
    chart_title.set_text(f'{label_text} - Phân tích biên độ CSI và phát hiện ranh giới chuyển động')

    t_data, s_data, s_idx, e_idx = data_source.generate_benchmark_pattern(pattern_name)
    s_ms = t_data[s_idx] if s_idx >= 0 else None
    e_ms = t_data[e_idx] if e_idx >= 0 else None

    new_opts = build_chart_options(t_data, s_data, s_ms, e_ms, y_min, y_max)
    chart_main.options.clear()
    chart_main.options.update(new_opts)
    chart_main.update()

    # Cập nhật Màn hình 2
    if pattern_name == "falling":
        status_icon.props('name=personal_injury').classes('text-red-600', remove='text-slate-400 text-blue-500 text-amber-500')
        status_title.set_text('CẢNH BÁO: PHÁT HIỆN NGÃ (FALLING)!').classes('text-red-600', remove='text-slate-600 text-blue-600 text-amber-600')
        conf_label.set_text('94.5%')
        conf_progress.set_value(0.945).props('color=red')
    elif pattern_name == "sitting_down":
        status_icon.props('name=chair').classes('text-amber-500', remove='text-slate-400 text-red-500 text-blue-500')
        status_title.set_text('CÓ NGƯỜI (NGỒI XUỐNG / SITTING DOWN)').classes('text-amber-600', remove='text-slate-600 text-red-600 text-blue-600')
        conf_label.set_text('92.0%')
        conf_progress.set_value(0.92).props('color=amber')
    elif pattern_name == "hand_swing":
        status_icon.props('name=front_hand').classes('text-indigo-500', remove='text-slate-400 text-red-500 text-amber-500')
        status_title.set_text('CÓ NGƯỜI (CỬ ĐỘNG TAY / HAND SWING)').classes('text-indigo-600', remove='text-slate-600 text-red-600 text-amber-600')
        conf_label.set_text('95.0%')
        conf_progress.set_value(0.95).props('color=indigo')
    else:
        status_icon.props('name=directions_walk').classes('text-red-500', remove='text-slate-400 text-amber-500 text-indigo-500')
        status_title.set_text('CÓ NGƯỜI (ĐANG DI CHUYỂN / WALKING)').classes('text-red-600', remove='text-slate-600 text-amber-600 text-indigo-600')
        conf_label.set_text('98.0%')
        conf_progress.set_value(0.98).props('color=red')

btn_walk.on_click(lambda: switch_pattern("walking", -0.025, 0.025, "(a) walking"))
btn_hand.on_click(lambda: switch_pattern("hand_swing", -0.025, 0.065, "(b) hand swing"))
btn_fall.on_click(lambda: switch_pattern("falling", -0.03, 0.035, "(c) falling"))
btn_sit.on_click(lambda: switch_pattern("sitting_down", -0.035, 0.045, "(d) sitting down"))

def start_realtime_stream():
    global current_mode
    current_mode = "realtime"
    active_activity_badge.set_text('Chế độ: Real-time Streaming')
    chart_title.set_text('📡 Luồng sóng CSI thời gian thực - Tự động bắt vạch Start / End')

btn_live.on_click(start_realtime_stream)

# ==================== REAL-TIME UPDATE LOOP ====================
def update_dashboard():
    if current_mode != "realtime":
        return

    res = data_source.step_realtime()
    t_data = [str(t) for t in res["time_ms"]]
    s_data = res["signal"]

    # Đánh dấu start/end nếu có
    s_ms = None
    e_ms = None
    if data_source.event_start_idx >= 0 and data_source.event_start_idx < len(t_data):
        s_ms = t_data[data_source.event_start_idx]
    if data_source.event_end_idx >= 0 and data_source.event_end_idx < len(t_data):
        e_ms = t_data[data_source.event_end_idx]

    new_opts = build_chart_options(t_data, s_data, s_ms, e_ms, -0.03, 0.03)
    chart_main.options.clear()
    chart_main.options.update(new_opts)
    chart_main.update()

    # Cập nhật Màn hình 2
    if res["is_event_active"]:
        region_badge.set_text('TRONG VÙNG HOẠT ĐỘNG (ACTIVE)').props('color=red-7')
        status_icon.props('name=directions_walk').classes('text-red-500 animate-pulse', remove='text-slate-400 text-blue-500')
        status_icon_wrap.classes('bg-red-50 dark:bg-red-950/40 border-2 border-red-500', remove='bg-slate-100 dark:bg-slate-800 border-0')
        status_title.set_text('CÓ NGƯỜI (ĐANG HOẠT ĐỘNG)').classes('text-red-600 dark:text-red-400', remove='text-slate-600 dark:text-slate-300')
        status_subtext.set_text(f'Phát hiện biến thiên biên độ vượt ngưỡng ({res["activity"]})')
        conf_progress.set_value(res["confidence"]).props('color=red')
        conf_label.set_text(f'{res["confidence"]*100:.1f}%').classes('text-red-500', remove='text-blue-500')
    else:
        region_badge.set_text('NGOÀI VÙNG HOẠT ĐỘNG (EMPTY)').props('color=grey-7')
        status_icon.props('name=person_off').classes('text-slate-400', remove='text-red-500 text-blue-500 animate-pulse')
        status_icon_wrap.classes('bg-slate-100 dark:bg-slate-800 border-0', remove='bg-red-50 dark:bg-red-950/40 border-2 border-red-500')
        status_title.set_text('PHÒNG TRỐNG').classes('text-slate-600 dark:text-slate-300', remove='text-red-600 dark:text-red-400')
        status_subtext.set_text('Tín hiệu CSI phẳng, không có biến động cơ thể người')
        conf_progress.set_value(0.97).props('color=blue')
        conf_label.set_text('97.0%').classes('text-blue-500', remove='text-red-500')

    log_table.rows = list(data_source.event_logs)
    log_table.update()

# Cập nhật định kỳ 100ms
ui.timer(0.1, update_dashboard)

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="Wi-Fi CSI Sensing - PMC11479185 Benchmark",
        port=8080,
        reload=False,
        show=True
    )
