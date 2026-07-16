"""add user_id to documents

Revision ID: 20260715_add_user_id
Revises: 5c2bea396263
Create Date: 2026-07-15 16:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20260715_add_user_id'
down_revision: Union[str, None] = '5c2bea396263'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Delete any existing documents first to ensure foreign key constraint works cleanly
    op.execute("DELETE FROM documents")
    
    # 1. Add user_id column
    op.add_column('documents', sa.Column('user_id', sa.Integer(), nullable=False))
    
    # 2. Create Foreign Key constraint
    op.create_foreign_key(
        'fk_documents_user_id_users',
        'documents', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE'
    )


def downgrade() -> None:
    # 1. Drop Foreign Key constraint
    op.drop_constraint('fk_documents_user_id_users', 'documents', type_='foreignkey')
    
    # 2. Drop user_id column
    op.drop_column('documents', 'user_id')
