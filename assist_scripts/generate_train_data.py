from comet_ml import end
from git import Optional
from numpy import save
from pathlib import Path
import yfinance as yf
import pandas as pd
import os


# --- Qlib 数据转换和处理步骤 ---
#
# 步骤 1: 数据采集 (当前脚本)
#   - 从 yfinance 获取原始股票数据。
#   - 对数据进行格式转换，以符合 Qlib 的要求。
#   - 将转换后的数据保存为 CSV 文件。
#
# 步骤 2: 转换为 Qlib 二进制格式 (需要手动执行)
#   - Qlib 为了高效读取，使用了自己的二进制数据格式。
#   - 在生成了所有股票的 CSV 文件后，需要使用 Qlib 提供的 `dump_bin.py` 工具进行转换。
#   - 命令行示例:
#     python <你的qlib路径>/scripts/dump_bin.py dump_all --csv_path <你的csv文件夹路径> --qlib_dir <你的qlib数据存储路径> --include_fields "open,high,low,close,vol,amt"
#
# 步骤 3: 在 Qlib 中使用
#   - 完成二进制转换后，就可以在 Qlib 的配置中指定数据路径，并进行后续的量化分析和模型训练。
#
# ------------------------------------

class Config:
    # Qlib 需要的字段
    QLIB_FEATURES = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
    # yfinance 提供的字段映射 (部分字段需要转换)
    YFINANCE_TO_QLIB_MAP = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        # 'Adj Close' 可以忽略，Qlib 不需要
    }
    QLIB_DATA_PATH = './qlib_data'  # Qlib 数据存储路径

    def __init__(self, data_name: str) -> None:
        # data_name 用于区分不同的数据集，例如 'cn_data_1y_1d' 表示中国市场，1年数据，日频。
        self.DATA_DIR = f'{self.QLIB_DATA_PATH}/{data_name}'
        # 从雅虎财经下载的股票数据会先转换为人类可读的 CSV 文件。
        self.CSV_SAVE_PATH = f'{self.DATA_DIR}/download/csvs'
        # CSV 会通过 qlib 的 dump_bin.py 脚本转换为二进制格式，供 qlib 高效读取。
        self.BIN_SAVE_PATH = f'{self.DATA_DIR}/download/bin'
        # 当能够获取到数据的 calendars、instruments、features 等信息后成为标准的训练数据集格式后，存放到 train 目录下。
        self.TRAIN_DATA_PATH = f'{self.DATA_DIR}/train'
        # 交易日历信息
        self.CALENDAR_PATH = f'{self.TRAIN_DATA_PATH}/calendars'
        # 股票列表信息
        self.INSTRUMENTS_PATH = f'{self.TRAIN_DATA_PATH}/instruments'
        # 特征列表信息
        self.FEATURES_PATH = f'{self.TRAIN_DATA_PATH}/features'

    def get_qlib_features(self) -> list[str]:
        return self.QLIB_FEATURES

    def get_data_dir(self):
        return self.DATA_DIR

    def get_csv_path(self):
        return self.CSV_SAVE_PATH

    def get_bin_path(self):
        return self.BIN_SAVE_PATH

    def get_train_path(self):
        return self.TRAIN_DATA_PATH

    def get_calendar_path(self):
        return self.CALENDAR_PATH

    def get_instruments_path(self):
        return self.INSTRUMENTS_PATH

    def get_features_path(self):
        return self.FEATURES_PATH


def convert_symbol_to_qlib(symbol):
    """
    将 yfinance 的股票代码格式 (如 600000.SS) 转换为 Qlib 格式 (如 SH600000)。

    :param symbol: str, yfinance 格式的股票代码。
    :return: str, Qlib 格式的股票代码。
    """
    parts = symbol.split('.')
    if len(parts) != 2:
        raise ValueError(f"股票代码格式不正确: {symbol}")

    stock_code, market = parts
    if market.upper() == 'SS':
        return f"SH{stock_code}"
    elif market.upper() == 'SZ':
        return f"SZ{stock_code}"
    else:
        raise ValueError(f"未知的市场后缀: {market}")

def get_index_hist_data(symbol, start_date, end_date, interval='1d') -> Optional[pd.DataFrame]:
    """
    获取指数的历史数据。

    :param symbol: str, yfinance 格式的指数代码 (例如 '^GSPC' 表示标普500指数)。
    :param start_date: str, 数据的起始日期 (格式 'YYYY-MM-DD')。
    :param end_date: str, 数据的结束日期 (格式 'YYYY-MM-DD')。
    :param interval: str, 数据的时间间隔 (例如 '1d', '1wk')。
    :return: pd.DataFrame or None, 返回包含指数历史数据的 DataFrame，如果获取失败则返回 None。
    """
    print(f"正在获取指数 {symbol} 的数据...")

    # 1. 创建指数对象并获取历史数据
    index = yf.Ticker(symbol)
    hist = index.history(start=start_date, end=end_date, interval=interval)

    # 2. 将索引 (Date) 转换为 'date' 列，并格式化
    hist.reset_index(inplace=True)
    # yfinance 返回的 'Date' 或 'Datetime' 列名可能带时区信息，统一处理
    date_col_name = hist.columns[0]
    hist.rename(columns={date_col_name: 'date'}, inplace=True)
    hist['date'] = pd.to_datetime(hist['date']).dt.strftime('%Y-%m-%d')

    if hist.empty:
        print(f"未能获取指数 {symbol} 的数据。请检查指数代码或网络连接。")
        return None

    print(f"成功获取指数 {symbol} 的数据，共 {len(hist)} 条记录。")

    return hist

def get_and_process_stock_data(symbol, save_path, start_date, end_date, interval='1d') -> Optional[pd.DataFrame]:
    """
    获取单个股票的数据，进行处理以符合 Qlib 格式，并保存为 CSV 文件。

    :param symbol: str, yfinance 格式的股票代码 (例如 '600000.SS')。
    :param start_date: str, 数据的起始日期 (格式 'YYYY-MM-DD')。
    :param end_date: str, 数据的结束日期 (格式 'YYYY-MM-DD')。
    :param interval: str, 数据的时间间隔 (例如 '1d', '1wk')。
    :param save_path: str, 保存 CSV 文件的目录。
    """

    # 1. 创建股票对象并获取历史数据
    hist = get_index_hist_data(symbol, start_date, end_date, interval)

    if hist is None or hist.empty:
        print(f"未能获取 {symbol} 的数据。请检查股票代码或网络连接。")
        return None

    # 2. 数据格式转换以符合 Qlib 要求
    print(f"正在处理 {symbol} 的数据...")

    # 2.1. 将列名转换为小写 (Open -> open)
    hist.rename(columns=str.lower, inplace=True)

    # 2.2. Qlib 需要 'volume' 和 'amount' 字段
    # 'amount' (成交额) 是通过 (开+高+低+收)/4 * 成交量 计算得出的
    hist['amount'] = (hist['open'] + hist['high'] + hist['low'] + hist['close']) / 4 * hist['volume']

    # 2.3. 选取 Qlib 需要的最终字段
    # 根据项目中的 `config.py`, 我们需要 ['open', 'high', 'low', 'close', 'volume', 'amount']
    qlib_features = config.get_qlib_features()
    hist = hist[qlib_features]

    # 3. 保存为 CSV 文件
    # 3.1. 转换股票代码为 Qlib 格式用于文件名
    qlib_symbol = convert_symbol_to_qlib(symbol)

    # 3.2. 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 3.3. 定义文件路径并保存
    file_path = os.path.join(save_path, f"{qlib_symbol}.csv")
    hist.to_csv(file_path, index=False)

    print(f"成功保存 {symbol} 的数据到 {file_path}")

    return hist


def generate_calendars(index_hist: Optional[pd.DataFrame], config: Config):
    if index_hist is None or index_hist.empty:
        print("未提供有效的历史数据，无法生成训练数据。")
        return

    # 生成交易日历
    trading_days = index_hist['date'].tolist()
    os.makedirs(config.get_calendar_path(), exist_ok=True)
    day_file: str = f"{config.get_calendar_path()}/day.txt"
    with open(day_file, 'w') as f:
        f.writelines("\n".join(trading_days))

    print(f"成功生成交易日历文件: {day_file}")

def generate_instruments(stock_list: list[str], config: Config):
    """
    根据给定的股票列表，自动获取上市日期并生成 Qlib 的 instruments 文件。

    :param stock_list: list, 包含 yfinance 格式股票代码的列表。
    :param save_path: str, 保存 instruments 文件的目录。
    """
    print("正在生成 instruments 文件...")
    instrument_lines = []

    for symbol in stock_list:
        try:
            print(f"正在获取 {symbol} 的上市日期...")

            # 1. 获取该股票的全部历史数据
            stock_hist = yf.Ticker(symbol).history(period='max', auto_adjust=False)

            if stock_hist.empty:
                print(f"警告：未能获取 {symbol} 的历史数据，已跳过。")
                continue

            # 2. 获取第一天的日期作为上市日期
            start_date = stock_hist.index[0].strftime('%Y-%m-%d')

            # 3. 设定一个遥远的未来日期作为退市日期
            end_date = '2200-12-31'

            # 4. 转换股票代码为 Qlib 格式
            qlib_symbol = convert_symbol_to_qlib(symbol)

            # 5. 格式化为 Qlib 所需的行
            line = f"{qlib_symbol}\t{start_date}\t{end_date}"
            instrument_lines.append(line)

        except Exception as e:
            print(f"处理 {symbol} 时发生错误: {e}")

    # 6. 将所有行写入文件
    os.makedirs(config.get_instruments_path(), exist_ok=True)
    file_path = os.path.join(config.get_instruments_path(), 'all.txt')

    with open(file_path, 'w') as f:
        for line in instrument_lines:
            f.write(line + '\n')

    print(f"成功生成 instruments 文件，已保存到: {file_path}")


if __name__ == "__main__":
    # --- 配置区 ---
    # 定义需要获取的股票列表 (使用 yfinance 格式)，为了正常生成交易日历，股票列表应同属一个市场。
    stock_list = [
        # "600519.SS",  # 贵州茅台
        # "000001.SZ",  # 平安银行
        "513060.SS",  # 博时恒生医疗
    ]
    config = Config(data_name='cn_data_1y_1d')
    start_date = '2021-03-29'
    end_date = '2024-12-31'

    # --- 执行区 ---
    ## 1. 获取并处理股票数据
    save_path: str = config.get_csv_path()
    if Path(save_path).exists():
        csv_list: list[str] = os.listdir(save_path)
    else:
        csv_list: list[str] = []
    index_hist: Optional[pd.DataFrame] = pd.DataFrame()
    if csv_list == []:
        print("未找到数据，开始下载...")
        for stock_symbol in stock_list:
            index_hist = get_and_process_stock_data(symbol=stock_symbol, save_path=save_path, start_date=start_date, end_date=end_date, interval='1d')
    else:
        for stock_symbol in stock_list:
            qlib_symbol = convert_symbol_to_qlib(stock_symbol)
            if f"{qlib_symbol}.csv" not in csv_list:
                index_hist = get_and_process_stock_data(symbol=stock_symbol, save_path=save_path, start_date=start_date, end_date=end_date, interval='1d')
            else:
                print(f"{qlib_symbol}.csv 已存在，跳过下载。")
                index_hist = get_index_hist_data(symbol=stock_symbol, start_date=start_date, end_date=end_date, interval='1d')

    ## 2. 生成交易日历文件 (day.txt)
    # generate_calendars(index_hist=index_hist, config=config)

    ## 3. 生成股票列表文件 (instruments/all.txt)
    # generate_instruments(stock_list=stock_list, config=config)

    ## 4. 生成特征列表文件 (features/all.txt)
    print(save_path)
    # python dump_bin.py dump_update --csv_path ./crypto_1min_upd --qlib_dir  ./crypto_1min_bin --freq 1min --date_field_name date --include_fields close
    os.popen(f"python ./qlib_scripts/dump_bin.py dump_all --data_path {save_path} --qlib_dir {config.get_train_path()} --freq day --include_fields 'open,high,low,close,volume,amount'")