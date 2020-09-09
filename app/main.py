import logging

logging.getLogger('cassandra').setLevel(logging.ERROR)
logging.getLogger('matplotlib.style.core').setLevel(logging.ERROR)

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from typing import Optional
import json
from detecting import Selector, find_communities, canvas_data
from functools import lru_cache
from cassandra import OperationTimedOut, UnresolvableContactPoints
import traceback
import tools

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

sels = {}


@app.get("/{snid}@{platform}/")
@lru_cache(maxsize=2)
def read_item(snid: int,
              platform: str,
              alg: str,
              params: Optional[str] = None):
    if snid <= 0:
        raise HTTPException(status_code=404,
                            detail="snid is not a valid user identifier")
    try:
        sel = sels.get(platform)
        if not sel:
            sel = Selector(platform)
            sels[platform] = sel
        nodes, edges = sel.ego_graph(int(snid))
    except (OperationTimedOut) as e:
        raise HTTPException(status_code=408,
                            detail="cassandra.OperationTimedOut")
    except UnresolvableContactPoints as e:
        raise HTTPException(status_code=503,
                            detail="cassandra.UnresolvableContactPoints")
    except (Exception) as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not edges:
        raise HTTPException(status_code=404, detail="Item not found")

    if params:
        params = params.split(',')
        params = {k: v for k, v in zip(params[::2], params[1::2])}

    try:
        cs = find_communities(len(nodes), edges, alg, params)
    except:
        msg = traceback.format_exc()
        tools.send_email(body=msg)
        logging.error(msg)
        raise HTTPException(status_code=500)

    if cs is None:
        raise HTTPException(status_code=501, detail="Not Implemented")

    data = canvas_data(nodes, edges, cs)
    data['snid'] = snid
    data['platform'] = platform

    return data