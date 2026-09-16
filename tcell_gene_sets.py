tcell_subtypes: dict[str, list[list[str]]] = {
    # Naive T cells
    'Naive': [
        ['CCR7', 'SELL', 'TCF7', 'LEF1', 'MAL', 'IL7R', 'LTB'],
        ['NKG7', 'CCL5', 'PRF1', 'GNLY', 'GZMB', 'FOXP3'],
    ],
    # Effector T cells
    'Cytotoxic': [
        ['CD8A', 'CD8B', 'NKG7', 'CCL5', 'CTSW', 'PRF1', 'GZMB', 'GZMH', 'FGFBP2'],
        ['CCR7', 'SELL', 'TCF7', 'LEF1', 'MAL', 'FOXP3'],
    ],
    'Th1': [
        ['CD4', 'TBX21', 'CXCR3', 'IFNG', 'IL12RB2', 'STAT4', 'CXCR6'],
        ['GATA3', 'IL4', 'IL5', 'IL13', 'RORC', 'IL17A', 'FOXP3'],
    ],
    'Th2': [
        ['CD4', 'GATA3', 'IL4', 'IL5', 'IL13', 'PTGDR2', 'CCR4'],
        ['TBX21', 'CXCR3', 'IFNG', 'RORC', 'IL17A', 'FOXP3'],
    ],
    'Tfh': [
        ['CD4', 'CXCR5', 'PDCD1', 'ICOS', 'BCL6', 'IL21', 'MAF', 'TOX2', 'SH2D1A'],
        ['FOXP3', 'KLRD1', 'GNLY', 'FGFBP2', 'GZMB'],
    ],
    'Th17': [
        ['CD4', 'RORC', 'CCR6', 'KLRB1', 'IL23R', 'IL17A', 'IL17F', 'CCL20'],
        ['TBX21', 'IFNG', 'GATA3', 'IL4', 'IL5', 'FOXP3'],
    ],
    'Treg': [
        ['CD4', 'FOXP3', 'IL2RA', 'CTLA4', 'IKZF2', 'TIGIT', 'TNFRSF18', 'LRRC32'],
        ['IL7R', 'IL2', 'CD40LG', 'IFNG', 'IL4', 'IL17A'],
    ],
    # Memory T cells
    'Memory central': [
        ['CCR7', 'SELL', 'IL7R', 'LTB', 'TCF7', 'LEF1', 'CD27', 'CD28'],
        ['FGFBP2', 'GNLY', 'GZMB', 'GZMH', 'PRF1', 'HAVCR2'],
    ],
    'Memory effector': [
        ['GZMK', 'CCL5', 'IL7R', 'CD27', 'LTB', 'CXCR3', 'CTSW'],
        ['CCR7', 'SELL', 'LEF1', 'FGFBP2', 'GNLY', 'GZMB'],
    ],
    # This represents a TRM-like transcriptional program in PBMCs,
    # not proof that the cells are genuinely tissue-resident.
    'Memory resident': [
        ['CD69', 'ITGA1', 'ITGAE', 'CXCR6', 'ZNF683', 'RGS1', 'RUNX3'],
        ['S1PR1', 'KLF2', 'SELL', 'CCR7', 'MAL'],
    ],
    # Exhaustion requires a combined program because individual checkpoint
    # genes can also be expressed during ordinary T-cell activation.
    'Exhausted': [
        ['TOX', 'PDCD1', 'TIGIT', 'LAG3', 'HAVCR2', 'ENTPD1', 'CTLA4', 'CXCL13'],
        ['CCR7', 'SELL', 'IL7R', 'TCF7', 'LEF1'],
    ],
}


"""Six proposed functional axes for human T-cell RNA-expression UMAP overlays.

These are hand-curated starting panels, not validated classifiers or complete
published signatures. Sets overlap intentionally. A within-dictionary winner
means the highest panel score under the chosen scoring method, not exclusive
biological identity. No scoring, binary thresholds, or integration code changes
are implemented here. Reuse expression-derived scores/labels across pre/post
embeddings when comparing the effect of integration.

Interpretation:
- Cytokine RNA indicates expression, not demonstrated secretion. TGFB1 requires
  extracellular activation; IL10/TGFB1 are neither exclusive nor obligatory Treg
  markers. Chemokines describe recruitment of other cells.
- Homing receptors describe potential interactions, not demonstrated migration.
  ITGA4/ITGB7 encode the two alpha4beta7 chains; an expression score does not test
  receptor assembly. CCR4 is not skin-specific. CD69 can indicate activation.
- pathway_enrichment contains compact expression panels. These are NOT complete
  MSigDB Hallmark sets; an expression mean is NOT a statistical enrichment test.
  Use full named sets and an explicit enrichment method for formal enrichment.
- Metabolic genes describe transport/enzyme expression, not metabolic flux.
  Nuclear OXPHOS genes are used instead of a mitochondrial-transcript QC score.
- TF expression does not measure DNA binding, nuclear localization, or activity.
  RORC totals cannot distinguish RORgamma from RORgammat. RUNX3 has CD8-lineage
  functions as well as residency roles. TOX/NR4A also occur outside exhaustion.
- Single-gene panels are intentionally left small; they are single-marker
  overlays, not multi-gene signatures. No unrelated genes are added as padding.

Evidence anchors (panel membership is proposed, not copied wholesale):
  Cytokine programs: Cherwinski 1987, doi:10.1084/jem.166.5.1229;
  Ivanov 2006, doi:10.1016/j.cell.2006.07.035; El-Behi 2011,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3116521/;
  Johnston 2009, doi:10.1126/science.1175870.
  Homing: Sallusto 1999, doi:10.1038/44385; Breitfeld 2000,
  doi:10.1084/jem.192.11.1545; Kumar 2017,
  doi:10.1016/j.celrep.2017.08.078; Mikhak 2013,
  https://rupress.org/jem/article/210/9/1855/45716/;
  Jonsson 2022, doi:10.1126/scitranslmed.abo0686.
  Pathway reference collections (full lists are larger than these panels):
  https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_INTERFERON_ALPHA_RESPONSE.html
  https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_TNFA_SIGNALING_VIA_NFKB.html
  https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_IL2_STAT5_SIGNALING.html
  https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_GLYCOLYSIS.html
  https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_OXIDATIVE_PHOSPHORYLATION.html
  Cytolytic program: Miller 2019, doi:10.1038/s41590-019-0312-6;
  Jonsson 2022, doi:10.1126/scitranslmed.abo0686.
  Metabolic interpretation: Raud 2018, PMID:30043753, experimentally separates
  Cpt1a expression/inhibition from requirements for T-cell differentiation.
  Adenosine: Deaglio 2007, doi:10.1084/jem.20062512.
  TF anchors: Hori 2003, doi:10.1126/science.1079490;
  Kanhere 2012, doi:10.1038/ncomms2260;
  Im 2016, doi:10.1038/nature19330; Khan 2019,
  doi:10.1038/s41586-019-1325-x; Kroenke 2012, PMID:22427637;
  Roychoudhuri 2016, doi:10.1038/ni.3441;
  Dominguez 2015, https://pmc.ncbi.nlm.nih.gov/articles/PMC4647261/;
  Milner 2017, PMID:29211713; Seo 2019, doi:10.1073/pnas.1905675116.
"""

secreted_cytokines: dict[str, list[str]] = {
    'Th1': [['IFNG', 'TNF', 'LTA'], []],
    'Th2': [['IL4', 'IL5', 'IL13'], []],
    'Th17': [['IL17A', 'IL17F', 'IL22'], []],
    'Treg': [['IL10', 'TGFB1'], []],
    'T cell growth': [['IL2'], []],
    'B cell help': [['IL21', 'IL4'], []],
    'Myeloid growth activation': [['CSF2', 'IL3'], []],
    'Inflammatory cell recruitment': [['CCL3', 'CCL4', 'CCL5', 'CCL20'], []],
    'DC recruitment': [['XCL1', 'XCL2'], []],
    'B cell recruitment': [['CXCL13'], []],
}

homing_molecules: dict[str, list[str]] = {
    'Lymph_node_entry': ['CCR7', 'SELL'],
    'Lymphoid_egress': ['S1PR1'],
    'Follicular_positioning': ['CXCR5'],
    'Inflamed_tissue_trafficking': ['CXCR3', 'CCR5', 'CCR2', 'CCR6'],
    'Gut_associated': ['CCR9', 'ITGA4', 'ITGB7'],
    'Skin_associated': ['CCR4', 'CCR10'],
    'Tissue_retention': ['CD69', 'ITGA1', 'ITGAE', 'CXCR6'],
    'Adhesion_transmigration': ['ITGAL', 'ITGB2', 'ITGA4', 'CX3CR1'],
}

# Compact programs for scoring; these are not complete Hallmark gene sets.
pathway_enrichment: dict[str, list[str]] = {
    'Cytolytic_effector_program': [
        'PRF1',
        'GZMB',
        'GZMH',
        'GNLY',
        'NKG7',
        'CTSW',
    ],
    'TCR_immediate_early_response': [
        'FOS',
        'JUNB',
        'EGR1',
        'EGR2',
        'NR4A1',
        'DUSP2',
    ],
    'Interferon_response': [
        'ISG15',
        'IFIT1',
        'IFIT2',
        'IFIT3',
        'MX1',
        'OAS1',
        'IFI6',
        'STAT1',
    ],
    'NFkB_response': [
        'NFKBIA',
        'TNFAIP3',
        'RELB',
        'BIRC3',
        'ICAM1',
        'TRAF1',
    ],
    'IL2_STAT5_associated_response': [
        'CISH',
        'SOCS2',
        'IL2RA',
        'BCL2',
        'PIM1',
    ],
    'Proliferation': [
        'MKI67',
        'TOP2A',
        'PCNA',
        'MCM2',
        'MCM5',
        'TYMS',
        'STMN1',
    ],
    'ER_stress_unfolded_protein_response': [
        'HSPA5',
        'HSP90B1',
        'HERPUD1',
        'DDIT3',
        'XBP1',
        'DNAJB9',
    ],
}

metabolic_signals: dict[str, list[str]] = {
    'Glucose_uptake_glycolysis': [
        'SLC2A1',
        'HK2',
        'PFKP',
        'PFKFB3',
        'ALDOA',
        'ENO1',
        'PKM',
        'LDHA',
    ],
    'Oxidative_phosphorylation': [
        'NDUFA9',
        'NDUFB8',
        'SDHA',
        'UQCRC1',
        'CYC1',
        'COX4I1',
        'COX5A',
        'ATP5F1A',
        'ATP5F1B',
    ],
    'Fatty_acid_oxidation_machinery': [
        'CPT1A',
        'CPT2',
        'SLC25A20',
        'ACADM',
        'ACADVL',
        'HADHA',
        'HADHB',
    ],
    'Fatty_acid_synthesis': [
        'ACLY',
        'ACACA',
        'FASN',
        'SCD',
        'ELOVL6',
    ],
    'Cholesterol_synthesis': [
        'HMGCS1',
        'HMGCR',
        'MVK',
        'MVD',
        'FDPS',
        'FDFT1',
        'SQLE',
    ],
    'Amino_acid_transport_glutamine_metabolism': [
        'SLC1A5',
        'SLC7A5',
        'SLC3A2',
        'GLS',
        'GLUD1',
        'GOT2',
    ],
    'Extracellular_adenosine_generation': ['ENTPD1', 'NT5E'],
    'Antioxidant_redox_defense': [
        'GCLC',
        'GCLM',
        'GSR',
        'TXN',
        'TXNRD1',
        'NQO1',
        'HMOX1',
    ],
}

transcription_factors: dict[str, list[str]] = {
    'Th1': [['TBX21'], []],
    'Th2': [['GATA3'], []],
    'Th17': [['RORC'], []],
    'Treg': [['FOXP3'], []],
    'Tfh': [['BCL6', 'MAF'], []],
    'Naive / memory maintenance': [['TCF7', 'LEF1', 'BACH2'], []],
    'Cytotoxic': [['TBX21', 'EOMES', 'RUNX3'], []],
    'Terminal effector': [['PRDM1', 'ZEB2', 'TBX21'], []],
    'Residence': [['RUNX3', 'PRDM1'], []],
    'Exhaustion': [
        ['TOX', 'TOX2', 'NR4A1', 'NR4A2', 'NR4A3'],
        [],
    ],
    'Immediate / early AP1': [['FOS', 'FOSB', 'JUN', 'JUNB'], []],
}


# Receptor plus selected intracellular partners and downstream signaling components.
# These are curated component panels, not validated receptor-activity signatures.
# Shared machinery can score without receptor expression under the unchanged scoring rule.
# Mouse mechanistic results are represented by human ortholog symbols; model limits are noted below.
costimulation: dict[str, list[str]] = {
    # TRAF2/5-NIK signaling plus the CD27-TRAF2-SHP-1 axis that modulates LCK in human naive T cells.
    # PTPN6 encodes SHP-1, a regulatory phosphatase; a higher panel score does not mean stronger TCR signaling.
    # Akiba 1998: https://doi.org/10.1074/jbc.273.21.13353
    # Jaeger-Ruckstuhl 2024: https://doi.org/10.1016/j.immuni.2024.01.011
    'CD27': [['CD27', 'TRAF2', 'TRAF5', 'MAP3K14', 'PTPN6', 'LCK'], []],
    # p85alpha-PI3K recruitment, GRB2-VAV1 signaling, and the LCK-PKCtheta connection.
    # PIK3R1 encodes p85alpha; PRKCQ encodes PKCtheta.
    # Fos 2008 (CD28/ICOS comparison): https://doi.org/10.4049/jimmunol.181.3.1969
    # Schneider 2008: https://pubmed.ncbi.nlm.nih.gov/18295596/
    # Kong 2011: https://doi.org/10.1038/ni.2120
    'CD28': [['CD28', 'LCK', 'PIK3R1', 'GRB2', 'VAV1', 'PRKCQ'], []],
    # PI3K recruitment and LCK/TBK1 association; PLCgamma1, RHOA and CDC42 support calcium/actin signaling.
    # PIK3R1 encodes both p50alpha and p85alpha; gene-level RNA does not distinguish these isoforms.
    # Fos 2008: https://doi.org/10.4049/jimmunol.181.3.1969
    # Pedros 2016: https://doi.org/10.1038/ni.3463
    # Leconte 2016: https://doi.org/10.1016/j.molimm.2016.09.022
    # Wan: https://doi.org/10.1038/s41423-018-0183-z
    'ICOS': [['ICOS', 'PIK3R1', 'LCK', 'TBK1', 'PLCG1', 'RHOA', 'CDC42'], []],
    # 4-1BB/CD137: TRAF1/2-cIAP machinery and the NIK-NFKB2/RELB alternative NF-kB branch.
    # BIRC2/BIRC3 encode cIAP1/2; MAP3K14 encodes NIK. These proteins also regulate basal signaling.
    # McPherson 2012 (mouse T cells): https://pubmed.ncbi.nlm.nih.gov/22570473/
    # Glez-Vaz 2023 (human and mouse): https://doi.org/10.1126/sciadv.adf6692
    'TNFRSF9': [
        [
            'TNFRSF9',
            'TRAF1',
            'TRAF2',
            'BIRC2',
            'BIRC3',
            'MAP3K14',
            'NFKB2',
            'RELB',
        ],
        [],
    ],
    # OX40: TRAF2/5 and the RIP1-PKCtheta-CARMA1/BCL10/MALT1-IKK signaling machinery.
    # CARD11 encodes CARMA1; CHUK/IKBKB/IKBKG encode IKKalpha/IKKbeta/NEMO.
    # Kawamata 1998: https://doi.org/10.1074/jbc.273.10.5808
    # So 2011 (T-cell hybridoma and primary mouse T cells): https://doi.org/10.1073/pnas.1008765108
    'TNFRSF4': [
        [
            'TNFRSF4',
            'TRAF2',
            'TRAF5',
            'RIPK1',
            'PRKCQ',
            'CARD11',
            'BCL10',
            'MALT1',
            'CHUK',
            'IKBKB',
            'IKBKG',
        ],
        [],
    ],
    # GITR: TRAF2/5, ERK1/2 (MAPK3/MAPK1), and canonical NF-kB components (NFKB1/RELA).
    # This pathway selection draws on mouse CD4/CD8 T-cell experiments.
    # Esparza 2006: https://doi.org/10.1074/jbc.M512915200
    # Snell 2010: https://doi.org/10.4049/jimmunol.1001912
    'TNFRSF18': [['TNFRSF18', 'TRAF2', 'TRAF5', 'MAPK1', 'MAPK3', 'NFKB1', 'RELA'], []],
    # DNAM-1: Src-family phosphorylation, GRB2-VAV1 coupling, p85-PI3K recruitment and PLCgamma1.
    # FYN is implicated in human T-cell signaling but is not uniquely required in mouse NK cells.
    # GRB2/PI3K/PLCG1 mechanistic evidence below is predominantly from NK-cell systems.
    # Shibuya 2003 (human naive T cells): https://pubmed.ncbi.nlm.nih.gov/14676297/
    # Zhang 2015 (mouse NK cells and human NK-cell line): https://doi.org/10.1084/jem.20150792
    'CD226': [['CD226', 'FYN', 'GRB2', 'VAV1', 'PIK3R1', 'PLCG1'], []],
}


# Expose the six functional dictionaries in analysis order.
FUNCTIONAL_GENE_SETS = {
    'secreted_cytokines': secreted_cytokines,
    'homing_molecules': homing_molecules,
    'pathway_enrichment': pathway_enrichment,
    'metabolic_signals': metabolic_signals,
    'transcription_factors': transcription_factors,
    'costimulation': costimulation,
}

# # The analysis expects section -> set -> gene -> regex for its name|symbol fields.
# MARKER_SETS = {
#     section_name: {
#         set_name: {gene: rf"(?:^|\|){gene}(?:\.\d+)?(?:\||$)" for gene in genes}
#         for set_name, genes in section.items()
#     }
#     for section_name, section in FUNCTIONAL_GENE_SETS.items()
# }

# # These panels have no identity gate; supply T-cell-filtered input for T-cell-only analysis.
# SECTION_SCOPES = {section_name: "all" for section_name in MARKER_SETS}

# # Every set participates in its own functional section's score comparison.
# WINNER_SETS = {section_name: list(section) for section_name, section in MARKER_SETS.items()}
