# Publication Artifact Boundaries

Datasets, split indexes, checkpoints, predictions, metrics, bootstrap outputs, generated
figures, and submission packages are not redistributed in this repository. They remain in
the controlled external artifact namespaces used by the study.

The following digests identify the frozen evidence without exposing local artifacts:

| Artifact | SHA-256 | Access state |
| --- | --- | --- |
| R6 validation freeze manifest | `c76f1738ba3cb0aee2d15b8c0d008c18321a0c7d729bc9c3feb66829458aa2c1` | `test_accessed=false` |
| R7 submission audit | `2a4288c23c0ba92217fed2c2d162ce24d2f6f8aace1def9e57e8f326762089cb` | `test_accessed=false` |
| R7 no-new-test revision freeze | `1b41cffd20ac3a4b8133b9616cad6638f74764e69d62d23814cd79ee9c7023bd` | `test_accessed=false`, `authorized=false` |

The historical RadioML 2016.10A test result is an earlier frozen whole-model comparison.
It cannot be replayed, sliced, re-bootstrapped, or used to support the post-test component
hypotheses in the controlled revision.

RadioML data must be obtained from its original source and handled according to the
license and provenance procedures in `docs/data/`.

Author names, contact details, affiliations, postal addresses, ORCID values, and journal
submission metadata are also excluded from the repository. The optional
`code/scripts/build_aeu_submission.py` utility requires this information through an
untracked external JSON file passed with `--author-metadata`. The required fields are
`name`, `short_name`, `email`, `organization`, `address_line`, `city`, `postal_code`,
`state`, `country`, `credit_statement`, and `submission_date`.
