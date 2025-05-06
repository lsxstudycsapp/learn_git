# Streamlit
## Streamlit开发文档
官方文档：[https://docs.streamlit.io/](https://docs.streamlit.io/)

中文文档：[https://blog.csdn.net/weixin_44458771/article/details/135495928](https://blog.csdn.net/weixin_44458771/article/details/135495928)



## Streamlit命令行启动
```powershell
pip install streamlit
streamlit run app.py --server.port 8501
```

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1722235586656-7a766396-565b-4c9b-868f-b343856378b3.png)

## <font style="color:rgb(51, 51, 51);">配置Pycharm调试Streamlit应用</font>
<font style="color:rgb(51, 51, 51);">开发环境</font>

<font style="color:rgb(51, 51, 51);">PyCharm Community Edition 2024</font>

<font style="color:rgb(51, 51, 51);">Win10/11</font>

<font style="color:rgb(51, 51, 51);">Streamlit </font><font style="color:#080808;background-color:#ffffff;">1.39.0</font>



### <font style="color:rgb(51, 51, 51);">创建应用</font>
<font style="color:rgb(51, 51, 51);">app.py</font>

```python
import streamlit as st

st.header("hello")
st.write("this is a streamlit demo")
```

### <font style="color:rgb(51, 51, 51);">启动应用</font>
<font style="color:rgb(51, 51, 51);">设置参数：</font>

<font style="color:rgb(51, 51, 51);">script：D:/software/python/python-3.12.4/.venv/Scripts/streamlit</font>

<font style="color:rgb(51, 51, 51);">script parameters：run app.py</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1722233961725-4bada41b-ea84-4d30-83e6-a92093fb6adf.png)

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1722224890183-20674e9f-c96c-416d-8207-94bde80d7e3a.png)

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1722234140695-5d1e769f-d197-41c2-bfc7-5acf77511461.png)

### <font style="color:rgb(51, 51, 51);">调试应用</font>
<font style="color:rgb(51, 51, 51);">点击调试按钮会报错。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1722234881001-b0d33b30-ba32-4ec3-9969-e51f25e40d00.png)

<font style="color:rgb(51, 51, 51);">经过分析，是因为选择的</font>`<font style="color:rgb(51, 51, 51);background-color:rgb(243, 244, 244);">script</font>`<font style="color:rgb(51, 51, 51);">, 文件名为</font>

`<font style="color:rgb(51, 51, 51);">D:/software/python/python-3.12.4/.venv/Scripts/</font><font style="color:rgb(51, 51, 51);background-color:rgb(243, 244, 244);">streamlit.exe</font>`

<font style="color:rgb(51, 51, 51);">streamlit.exe 是一个二进制文件，导致字符集解析出错。在这里把 linux 环境下的 streamlit 复制过来。</font>

```plain
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import re
import sys
from streamlit.web.cli import main
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
```

<font style="color:rgb(51, 51, 51);">复制到 D:/software/python/python-3.12.4/.venv/Scripts/</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1722234760136-375ad9f0-25d6-461c-8f20-556033ef1aa9.png)



<font style="color:rgb(51, 51, 51);">再次启动 debug 按钮，报错如下</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1722235154306-6f36b19b-3fc8-479f-a5de-53a6733d49eb.png)



<font style="color:rgb(51, 51, 51);">解决如下:</font>

<font style="color:rgb(51, 51, 51);">Help | Find Action | Registry | python.debug.asyncio.repl 去掉勾。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1722224890604-a51e91dd-bbfb-4930-9a99-f21d6c66a14a.png)

<font style="color:rgb(51, 51, 51);">再次点击 debug 就可以正常调试了。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1722235056650-c7f471d9-b801-4d0a-b80d-f01613bfd3e0.png)

  
 

