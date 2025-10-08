#Built in search Tool

from langchain_community.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()
results = search_tool.invoke('top news in india today')
print(results)



#Built in Shell Tool

from langchain_community.tools import ShellTool
shell_tool = ShellTool()
results = shell_tool.invoke('ls')
print(results)