import aiohttp
import asyncio

async def fetch(client:aiohttp.ClientSession, url, text):
  async with client.post(url, data={'text':text}) as resp:
    return await resp.json()
  
async def concurrent_get(url, lst):
  async with aiohttp.ClientSession() as client:
    tasks = []
    for item in lst:
      tasks.append(asyncio.create_task(fetch(client, url, item)))
    return await asyncio.wait(tasks)
  
  
# 并发网络访问
url = 'www.example.com'
lst = ['item1', 'item2', ...]
loop = asyncio.get_event_loop()
res = loop.run_until_complete(concurrent_get(url, lst)
    
