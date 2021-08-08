

# technical indicator functions

def calc_sma(data, window):
    '''
    simple moving average
    '''
    sma = data['close'].rolling(window).mean()
    return sma
        
def calc_bb_width(data, window, scale=2):
    '''
    bollinger bands width
    '''
    bb_width = data['close'].rolling(window).std() * scale
    return bb_width

def calc_ror(data, periods):
    '''
    rate of return (pct change over period)
    '''
    ror = data['close'].pct_change(periods=periods, fill_method='ffill')
    return ror 

if __name__ == '__main__':
    pass