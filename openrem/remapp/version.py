__version__ = "1.0.0b1"  # Actual version number
__docs_version__ = "1.0.0b1-docs"  # Should match the branch readthedocs will build against - tag + '-docs'
__short_version__ = "1.0"  # Short version number for setuptools config
__skin_map_version__ = "0.9.0"  # To enable changes to skinmap file format
__netdicom_implementation_version__ = (
    "1.0.0.0"  # Used as part of UID when storing DICOM objects
)
# IANA OID for OpenREM, + 1 + major version + minor + patch + iteration (increment for betas if pydicom code changes)
__implementation_uid__ = "1.3.6.1.4.1.45593.1.1.0.0.1"
# IANA private enterprise number for OpenREM, + 2
__openrem_root_uid__ = "1.3.6.1.4.1.45593.2."
