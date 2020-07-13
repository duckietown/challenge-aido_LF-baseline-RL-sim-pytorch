
update-reqs: # todo
	pur --index-url $(PIP_INDEX_URL) -r requirements.txt -f -m '*' -o requirements.resolved
	aido-update-reqs requirements.resolved

submit: update-reqs
	dts challenges submit

submit-bea: update-reqs
	dts challenges submit --impersonate 1639 --challenge all --retire-same-label
