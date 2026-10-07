import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('training_report',Path(__file__).parents[1]/'scripts/report_training.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def test_report_handles_live_partial_row_and_rejects_invalid_metrics(tmp_path):
    row={"step":1,"demo":.5,"action":.4,"feature":.3,"total":.93,"grad_norm":2.,"seconds":.1}
    p=tmp_path/'metrics.jsonl'
    p.write_text(json.dumps(row)+'\n{"step":')
    (tmp_path/'validation.jsonl').write_text(json.dumps({'step':0,'validation_l1':.6})+'\n')
    m=report.render(tmp_path)
    assert m['last_step']==1
    assert (tmp_path/'curves.png').stat().st_size>1000
    assert '机器人任务成功率' in (tmp_path/'report.html').read_text()
    p.write_text(json.dumps({**row,'total':float('nan')})+'\n')
    with pytest.raises(ValueError,match='Non-finite'):report.render(tmp_path)
    p.write_text((json.dumps(row)+'\n')*2)
    with pytest.raises(ValueError,match='Duplicate'):report.render(tmp_path)
