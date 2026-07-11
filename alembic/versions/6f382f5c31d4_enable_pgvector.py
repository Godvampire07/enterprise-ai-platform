"""enable pgvector

Revision ID: 6f382f5c31d4
Revises: 61b68090edb0
Create Date: 2026-07-10 15:40:11.996243

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6f382f5c31d4'
down_revision: Union[str, None] = '61b68090edb0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")
