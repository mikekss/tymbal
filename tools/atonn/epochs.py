#!/usr/bin/env python3
"""epochs.py — static parsing of the generated network.c: epoch block types and
the compiler's cost estimate. Needed to compare graph compilation VARIANTS
locally, without tying up the board."""
import io,re,sys
p=sys.argv[1]
s=io.open(p,encoding='utf-8',errors='replace').read()
i=s.index('ll_atonn_rt_epoch_block_array[] = {')
j=s.index('\n  };', i)
items=re.findall(r'\{(.*?)\n\s*\},', s[i:j], re.S)
rows=[]
for it in items:
    g=lambda k:(int(m.group(1)) if (m:=re.search(r'\.%s\s*=\s*([0-9]+)'%k,it)) else 0)
    f=(m.group(1) if (m:=re.search(r'\.flags\s*=\s*([^\n]+)',it)) else '')
    kind='hybrid' if 'hybrid' in f else ('EC' if 'blob' in f else ('pure_hw' if 'pure_hw' in f else ('pure_sw' if 'pure_sw' in f else '?')))
    rows.append((g('epoch_num'),g('last_epoch_num'),kind,g('estimated_npu_cycles'),g('estimated_tot_cycles')))
agg={}
for e0,e1,k,n,t in rows:
    a=agg.setdefault(k,[0,0,0]); a[0]+=1; a[1]+=n; a[2]+=t
tot=sum(r[4] for r in rows)
print('%-22s blocks=%-3d  total estimate=%d' % (p.split('/')[-2] if '/' in p else p, len(rows), tot))
for k,(c,n,t) in sorted(agg.items(), key=lambda x:-x[1][2]):
    print('    %-8s %3d pcs  npu=%-9d tot=%-9d' % (k,c,n,t))
big=[r for r in sorted(rows,key=lambda r:-r[4]) if r[2]=='hybrid'][:6]
if big: print('    most expensive hybrids:', ', '.join('ep%d:%d'%(r[0],r[4]) for r in big))
