"""Fetch CVE write-ups as plain text (curl + html2text), so the stored text is the real article
rather than a model's summary. Skips files that already exist."""
import os, re, subprocess, sys
import html2text
from bs4 import BeautifulSoup

ART = {  # name: (url, class, role)
 "shiftfs":        ("https://www.synacktiv.com/publications/exploitation-of-a-double-free-vulnerability-in-ubuntu-shiftfs-driver-cve-2021-3492.html", "uaf", "train"),
 "ios_voucher":    ("https://www.synacktiv.com/en/publications/analysis-and-exploitation-of-the-ios-kernel-vulnerability-cve-2021-1782.html", "uaf", "train"),
 "looney_tunables":("https://www.qualys.com/2023/10/03/cve-2023-4911/looney-tunables-local-privilege-escalation-glibc-ld-so.txt", "heap_overflow", "train"),
 "netfilter_22555":("https://google.github.io/security-research/pocs/linux/cve-2021-22555/writeup.html", "heap_overflow", "train"),
 "sequoia":        ("https://www.qualys.com/2021/07/20/cve-2021-33909/sequoia-local-privilege-escalation-linux.txt", "integer_overflow", "train"),
 "win_tcpip":      ("https://malwaretech.com/2024/08/exploiting-CVE-2024-38063.html", "integer_overflow", "train"),
 "confluence_ognl":("https://github.com/httpvoid/writeups/blob/main/Confluence-RCE.md", "command_injection", "train"),
 "spring4shell":   ("https://www.lunasec.io/docs/blog/spring-rce-vulnerabilities/", "command_injection", "train"),
 "exiftool":       ("https://devcraft.io/2021/05/04/exiftool-arbitrary-code-execution-cve-2021-22204.html", "command_injection", "train"),
 "dirtycow":       ("https://github.com/firefart/dirtycow", "race_condition", "train"),
 "vsstandard":     ("https://www.mdsec.co.uk/2024/01/cve-2024-20656-local-privilege-escalation-in-vsstandardcollectorservice150-service/", "race_condition", "train"),
 "ios_machmsg":    ("https://www.synacktiv.com/en/publications/ios-1-day-hunting-uncovering-and-exploiting-cve-2020-27950-kernel-memory-leak.html", "uninit_memory", "train"),
}

ART.update({
 "nftables":         ("https://pwning.tech/nftables/", "uaf", "holdout"),
 "cve_2022_0185":    ("https://www.willsroot.io/2022/01/cve-2022-0185.html", "heap_overflow", "train"),
 "ghostscript_28879":("https://offsec.almond.consulting/ghostscript-cve-2023-28879.html", "heap_overflow", "train"),
 "win32k_1732":      ("https://googleprojectzero.github.io/0days-in-the-wild//0day-RCAs/2021/CVE-2021-1732.html", "oob_access", "train"),
 "dirtycow":         ("https://chao-tic.github.io/blog/2017/05/24/dirty-cow", "race_condition", "train"),
})

def to_text(raw, url):
    if "<html" not in raw[:2000].lower() and "<!doctype" not in raw[:200].lower():
        return raw                                   # already plain text (qualys .txt)
    soup = BeautifulSoup(raw, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        t.decompose()
    main = soup.find("article") or soup.find("main") or soup.find(id="content") or soup.body or soup
    h = html2text.HTML2Text(); h.ignore_links = True; h.ignore_images = True; h.body_width = 0
    txt = h.handle(str(main))
    return re.sub(r"\n{3,}", "\n\n", txt).strip()

os.makedirs("data/jlens/cve", exist_ok=True)
for name, (url, cls, role) in ART.items():
    out = f"data/jlens/cve/{name}.txt"
    if os.path.exists(out):
        print(f"  {name:17} exists, skipping"); continue
    try:
        raw = subprocess.run(["curl", "-sSL", "--max-time", "45", "-A", "Mozilla/5.0", url],
                             capture_output=True, text=True, timeout=60).stdout
        txt = to_text(raw, url)
        if len(txt) < 1500:
            print(f"  {name:17} TOO SHORT ({len(txt)} chars) -- needs manual attention"); continue
        open(out, "w", encoding="utf-8").write(txt)
        print(f"  {name:17} {len(txt):>7} chars  [{cls}]")
    except Exception as e:
        print(f"  {name:17} FAILED: {e}")
