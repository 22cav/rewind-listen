"""The VoiceBench split used for every number: each item goes to dev (30 %) or test (70 %) by a hash of its id alone,
so the split is the same on every machine and run. Item ids are "<subset file stem>/<row>", e.g.
"openbookqa/test/0" or "mmsu/biology/12" (the VoiceBench parquet files of hlt-lab/voicebench)."""
import hashlib

DEV_SHARE = 0.3


def split_of(item_id: str) -> str:
    """'dev' or 'test', fixed by the item id."""
    h = int(hashlib.sha1(item_id.encode()).hexdigest()[:8], 16) / 16**8
    return "dev" if h < DEV_SHARE else "test"
